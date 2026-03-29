"""
MoodVerse — Flask Backend
Movies + Music + Anime recommendation platform
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3, json, os, secrets, re, threading
from functools import wraps

from database import init_db, get_conn
from recommend import (get_recommendations, get_music_recommendations,
                       get_anime_recommendations, get_ncf_recommendations,
                       predict_sentiment, predict_genres,
                       detect_mood_from_text, PROCESSED_DIR,
                       load_sentiment_model, load_genre_model,
                       get_similar_movies, get_similar_music, get_similar_anime)

app = Flask(__name__, static_folder='.')
CORS(app)

# ── Missing File Validator (For GitHub/Drive Distribution) ───────────────────
REQUIRED_RESOURCE_FILES = [
    'models/svd_artifacts.pkl',
    'datasets/raw/anime.csv',
    'datasets/raw/spotify_tracks.csv'
]

def get_missing_files():
    return [os.path.basename(f) for f in REQUIRED_RESOURCE_FILES if not os.path.exists(f)]

@app.route('/api/system_status', methods=['GET'])
def system_status():
    missing = get_missing_files()
    return jsonify({
        'ready': len(missing) == 0,
        'missing': missing
    })


# ── Analytics & Poster caches (pre-loaded or dynamic) ────────────────────────
_analytics_cache = None
_analytics_lock  = threading.Lock()
# RAM caches (fast) + SQLite (persistent)
_poster_cache    = {}  
_anime_cache     = {}  
_cache_lock      = threading.Lock()

def _build_analytics_cache():
    """Pre-aggregate analytics stats once at startup to avoid per-request CSV loading."""
    global _analytics_cache
    try:
        import pandas as pd
        ratings    = pd.read_csv(os.path.join(PROCESSED_DIR, 'ratings_clean.csv'))
        movies     = pd.read_csv(os.path.join(PROCESSED_DIR, 'movies_clean.csv'))
        movies_exp = pd.read_csv(os.path.join(PROCESSED_DIR, 'movies_exploded.csv'))

        genre_counts = movies_exp['genre'].value_counts().head(15)
        genre_data   = [{'genre': g, 'count': int(c)} for g, c in genre_counts.items()]

        rating_dist = ratings['rating'].value_counts().sort_index()
        rating_data = [{'rating': float(r), 'count': int(c)} for r, c in rating_dist.items()]

        top_movies = (
            ratings.groupby('movieId')
            .agg(avg_rating=('rating','mean'), count=('rating','count'))
            .reset_index()
            .merge(movies[['movieId','title']], on='movieId')
            .query('count >= 200')
            .sort_values('avg_rating', ascending=False)
            .head(10)
        )
        top_data = [
            {'title': re.sub(r'\s*\(\d{4}\)\s*$', '', r['title']).strip(),
             'avg_rating': round(r['avg_rating'], 2), 'count': int(r['count'])}
            for _, r in top_movies.iterrows()
        ]

        genre_ratings = (
            ratings.merge(movies_exp[['movieId','genre']], on='movieId', how='left')
            .groupby('genre')['rating'].mean()
            .sort_values(ascending=False).head(15)
        )
        genre_rating_data = [{'genre': g, 'avg_rating': round(float(v), 2)}
                              for g, v in genre_ratings.items()]

        summary = {
            'total_ratings': int(len(ratings)),
            'total_movies':  int(movies['movieId'].nunique()),
            'total_users':   int(ratings['userId'].nunique()),
            'mean_rating':   round(float(ratings['rating'].mean()), 3)
        }

        with _analytics_lock:
            _analytics_cache = {
                'success':      True,
                'summary':      summary,
                'genre_counts': genre_data,
                'rating_dist':  rating_data,
                'top_movies':   top_data,
                'genre_ratings': genre_rating_data
            }
        print('Analytics cache built successfully.')
    except Exception as e:
        print(f'Analytics pre-load error: {e}')
        with _analytics_lock:
            _analytics_cache = {'success': False, 'error': str(e)}

import bcrypt

def hash_password(pw):
    return bcrypt.hashpw(pw.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(plain_pw, hashed_pw):
    return bcrypt.checkpw(plain_pw.encode('utf-8'), hashed_pw.encode('utf-8'))

def generate_token(uid):
    token = secrets.token_hex(32)
    conn = get_conn()
    conn.execute('INSERT INTO sessions (token, user_id) VALUES (?, ?)', (token, uid))
    conn.commit()
    conn.close()
    return token

def get_user_from_token(token):
    if not token: return None
    conn = get_conn()
    res = conn.execute('SELECT user_id FROM sessions WHERE token=?', (token,)).fetchone()
    conn.close()
    return res['user_id'] if res else None

def auth_required(f):
    @wraps(f)
    def decorated(*args,**kwargs):
        token=request.headers.get('Authorization','').replace('Bearer ','')
        uid=get_user_from_token(token)
        if not uid: return jsonify({'success':False,'error':'Not authenticated.'}),401
        return f(*args,user_id=uid,**kwargs)
    return decorated


# ── Frontend ──────────────────────────────────────────────────────────────
@app.route('/')
def index(): return send_from_directory('.','index.html')


# ── Auth ──────────────────────────────────────────────────────────────────
@app.route('/api/auth/register', methods=['POST'])
def register():
    data=request.get_json(force=True)
    username=data.get('username','').strip()
    email=data.get('email','').strip()
    password=data.get('password','')
    if not username or not email or not password:
        return jsonify({'success':False,'error':'All fields required.'}),400
    if len(password)<6:
        return jsonify({'success':False,'error':'Password must be at least 6 characters.'}),400
    if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email):
        return jsonify({'success':False,'error':'Please enter a valid email address.'}),400
    try:
        conn=get_conn()
        conn.execute('INSERT INTO users (username,email,password) VALUES (?,?,?)',
                     (username,email,hash_password(password)))
        conn.commit()
        uid=conn.execute('SELECT id FROM users WHERE username=?',(username,)).fetchone()['id']
        conn.close()
        token=generate_token(uid)
        return jsonify({'success':True,'token':token,'user':{'id':uid,'username':username,'email':email}})
    except sqlite3.IntegrityError:
        return jsonify({'success':False,'error':'Username or email already exists.'}),400

@app.route('/api/auth/login', methods=['POST'])
def login():
    data=request.get_json(force=True)
    username=data.get('username','')
    password=data.get('password','')
    conn=get_conn()
    user=conn.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
    conn.close()
    if not user or not check_password(password, user['password']):
        return jsonify({'success':False,'error':'Invalid username or password.'}),401
    token=generate_token(user['id'])
    return jsonify({'success':True,'token':token,
                    'user':{'id':user['id'],'username':user['username'],'email':user['email']}})

@app.route('/api/auth/logout', methods=['POST'])
@auth_required
def logout(user_id):
    token=request.headers.get('Authorization','').replace('Bearer ','')
    conn = get_conn()
    conn.execute('DELETE FROM sessions WHERE token=?', (token,))
    conn.commit()
    conn.close()
    return jsonify({'success':True})

@app.route('/api/auth/me', methods=['GET'])
@auth_required
def me(user_id):
    conn=get_conn()
    user=conn.execute('SELECT id,username,email,created_at FROM users WHERE id=?',(user_id,)).fetchone()
    conn.close()
    if not user: return jsonify({'success':False,'error':'User not found.'}),404
    return jsonify({'success':True,'user':dict(user)})


# ── Movies ────────────────────────────────────────────────────────────────
@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data=request.get_json(force=True)
        movies=get_recommendations(
            genres=data.get('genres',['Drama']),
            energy=int(data.get('energy',50)), darkness=int(data.get('darkness',20)),
            romance=int(data.get('romance',30)), adventure=int(data.get('adventure',50)),
            decade_filter=data.get('decade_filter'), genre_filter=data.get('genre_filter'), n=10)
        _save_rec_history(request, data, movies)
        return jsonify({'success':True,'movies':movies})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500



# ── NCF (Neural Collaborative Filtering) ─────────────────────────────────
@app.route('/api/recommend/ncf', methods=['POST'])
def recommend_ncf():
    try:
        data  = request.get_json(force=True)
        mood  = data.get('mood', 'adventurous')
        genres= data.get('genres', [])
        movies= get_ncf_recommendations(mood=mood, genres=genres or None, n=10)
        _save_rec_history(request, data, movies)
        return jsonify({'success': True, 'movies': movies, 'model': 'NCF — NeuMF (GMF + MLP)'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Music ─────────────────────────────────────────────────────────────────
@app.route('/api/recommend/music', methods=['POST'])
def recommend_music():
    try:
        data=request.get_json(force=True)
        tracks=get_music_recommendations(
            mood=data.get('mood','chill'),
            genre_filter=data.get('genre_filter'), n=10)
        return jsonify({'success':True,'tracks':tracks})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500


# ── Anime ─────────────────────────────────────────────────────────────────
@app.route('/api/recommend/anime', methods=['POST'])
def recommend_anime():
    try:
        data=request.get_json(force=True)
        anime=get_anime_recommendations(
            mood=data.get('mood','adventurous'),
            genre_filter=data.get('genre_filter'), n=10)
        return jsonify({'success':True,'anime':anime})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500


# ── Similarity Search ─────────────────────────────────────────────────────
@app.route('/api/similar/movies', methods=['POST'])
def similar_movies():
    """Find movies similar to a given title using NCF embeddings or TF-IDF."""
    try:
        data  = request.get_json(force=True)
        title = data.get('title', '').strip()
        n     = int(data.get('n', 10))
        if not title:
            return jsonify({'success': False, 'error': 'No title provided.'}), 400
        result = get_similar_movies(title, n=n)
        if not result['found']:
            return jsonify({'success': False, 'error': result.get('error', 'Not found.')}), 404
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/similar/music', methods=['POST'])
def similar_music():
    """Find tracks similar to a given track name using audio feature cosine similarity."""
    try:
        data  = request.get_json(force=True)
        title = data.get('title', '').strip()
        n     = int(data.get('n', 10))
        if not title:
            return jsonify({'success': False, 'error': 'No title provided.'}), 400
        result = get_similar_music(title, n=n)
        if not result['found']:
            return jsonify({'success': False, 'error': result.get('error', 'Not found.')}), 404
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/similar/anime', methods=['POST'])
def similar_anime():
    """Find anime similar to a given title using TF-IDF genre cosine similarity."""
    try:
        data  = request.get_json(force=True)
        title = data.get('title', '').strip()
        n     = int(data.get('n', 10))
        if not title:
            return jsonify({'success': False, 'error': 'No title provided.'}), 400
        result = get_similar_anime(title, n=n)
        if not result['found']:
            return jsonify({'success': False, 'error': result.get('error', 'Not found.')}), 404
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# -- AI Mood (NLP TF-IDF Mood Detection -- local ML, no API key needed) -------
@app.route('/api/interpret-mood', methods=['POST'])
def interpret_mood():
    """
    Detect mood from free text using TF-IDF cosine similarity over a
    mood-labelled corpus (sklearn NLP). Optionally upgrades to Anthropic
    Claude if ANTHROPIC_API_KEY env var is set.
    Demonstrates real AIML domain: NLP + Vector Space Models.
    """
    try:
        text=request.get_json(force=True).get('text','').strip()
        if not text: return jsonify({'success':False,'error':'No text provided.'}),400

        # Try Anthropic only if a real key is set in environment
        api_key = os.environ.get('ANTHROPIC_API_KEY','')
        if api_key and api_key not in ('','PASTE_YOUR_ANTHROPIC_KEY_HERE'):
            try:
                import anthropic
                client=anthropic.Anthropic(api_key=api_key)
                response=client.messages.create(
                    model='claude-3-haiku-20240307', max_tokens=400,
                    messages=[{'role':'user','content':f"""Analyze mood, return ONLY JSON:
{{"mood_label":"<happy|melancholic|thrilled|romantic|adventurous|nostalgic|scared|inspired|chill|dark>",
"genres":["<2-4 from: Action,Adventure,Animation,Comedy,Crime,Documentary,Drama,Family,Fantasy,Film-Noir,Horror,Musical,Mystery,Romance,Sci-Fi,Thriller,War,Western>"],
"energy":<0-100>,"darkness":<0-100>,"romance":<0-100>,"adventure":<0-100>,
"explanation":"<one warm sentence>"}}
User mood: "{text}"
"""}])
                result=json.loads(response.content[0].text)
                result['model_used']='Claude (Anthropic LLM)'
                return jsonify({'success':True,**result})
            except Exception:
                pass  # fall through to local NLP

        # Local NLP: TF-IDF + cosine similarity mood detection (sklearn)
        result = detect_mood_from_text(text)
        if not result.get('success'):
            return jsonify({'success':False,'error':result.get('error','NLP model failed.')}),500
        return jsonify(result)
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}),500



# ── Sentiment ─────────────────────────────────────────────────────────────
@app.route('/api/sentiment', methods=['POST'])
def sentiment():
    try:
        text=request.get_json(force=True).get('text','').strip()
        if not text or len(text)<10: return jsonify({'success':False,'error':'Review too short.'}),400
        result=predict_sentiment(text)
        token=request.headers.get('Authorization','').replace('Bearer ','')
        uid=get_user_from_token(token)
        if uid and 'error' not in result:
            conn=get_conn()
            conn.execute('INSERT INTO review_history (user_id,review_text,sentiment,confidence) VALUES (?,?,?,?)',
                         (uid,text[:500],result['sentiment'],result['confidence']))
            conn.commit(); conn.close()
        return jsonify({'success':True,**result})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500


# ── Genre Predictor ───────────────────────────────────────────────────────
@app.route('/api/predict-genres', methods=['POST'])
def genre_predict():
    try:
        data=request.get_json(force=True)
        title=data.get('title','').strip()
        if not title: return jsonify({'success':False,'error':'No title provided.'}),400
        result=predict_genres(title, release_year=int(data.get('release_year',2000)))
        return jsonify({'success':True,**result})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500


# ── Watchlist ─────────────────────────────────────────────────────────────
@app.route('/api/watchlist', methods=['GET'])
@auth_required
def get_watchlist(user_id):
    conn=get_conn()
    items=conn.execute('SELECT * FROM watchlist WHERE user_id=? ORDER BY added_at DESC',(user_id,)).fetchall()
    conn.close()
    return jsonify({'success':True,'watchlist':[dict(i) for i in items]})

@app.route('/api/watchlist/add', methods=['POST'])
@auth_required
def add_watchlist(user_id):
    data=request.get_json(force=True)
    try:
        conn=get_conn()
        conn.execute('INSERT OR IGNORE INTO watchlist (user_id,movie_id,title,genres,avg_rating,release_year,poster) VALUES (?,?,?,?,?,?,?)',
                     (user_id,data.get('movieId'),data.get('title'),json.dumps(data.get('genres',[])),
                      data.get('avg_rating'),data.get('release_year'),data.get('poster')))
        conn.commit(); conn.close()
        return jsonify({'success':True})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/watchlist/remove/<int:movie_id>', methods=['DELETE'])
@auth_required
def remove_watchlist(user_id,movie_id):
    conn=get_conn()
    conn.execute('DELETE FROM watchlist WHERE user_id=? AND movie_id=?',(user_id,movie_id))
    conn.commit(); conn.close()
    return jsonify({'success':True})


# ── Analytics (served from pre-built cache) ───────────────────────────────
@app.route('/api/analytics/overview', methods=['GET'])
def analytics_overview():
    with _analytics_lock:
        cached = _analytics_cache
    if cached is None:
        return jsonify({'success': False, 'error': 'Analytics still loading, try again shortly.'}), 503
    return jsonify(cached)

@app.route('/api/analytics/user', methods=['GET'])
@auth_required
def user_analytics(user_id):
    conn=get_conn()
    wl=conn.execute('SELECT COUNT(*) as c FROM watchlist WHERE user_id=?',(user_id,)).fetchone()['c']
    rv=conn.execute('SELECT COUNT(*) as c FROM review_history WHERE user_id=?',(user_id,)).fetchone()['c']
    rc=conn.execute('SELECT COUNT(*) as c FROM rec_history WHERE user_id=?',(user_id,)).fetchone()['c']
    reviews=conn.execute('SELECT sentiment,COUNT(*) as c FROM review_history WHERE user_id=? GROUP BY sentiment',(user_id,)).fetchall()
    conn.close()
    return jsonify({'success':True,'watchlist_count':wl,'reviews_analyzed':rv,
                    'recommendations_run':rc,'sentiment_breakdown':{r['sentiment']:r['c'] for r in reviews}})


# ── OMDB Poster Proxy (avoids CORS / key issues client-side) ─────────────────
@app.route('/api/poster', methods=['GET'])
def poster_proxy():
    """Fetch movie poster and plot from OMDB API server-side with persistent SQLite caching."""
    title = request.args.get('title', '').strip()
    year  = request.args.get('year', '').strip()
    if not title:
        return jsonify({'success': False, 'error': 'No title provided'}), 400
    
    cache_key = f"{title.lower()}_{year}"
    
    # 1. Check RAM cache first
    with _cache_lock:
        if cache_key in _poster_cache:
            return jsonify(_poster_cache[cache_key])

    # 2. Check SQLite persistent cache
    try:
        conn = get_conn()
        res = conn.execute('SELECT * FROM poster_cache WHERE key = ?', (cache_key,)).fetchone()
        conn.close()
        if res:
            data = {
                'success': bool(res['success']),
                'poster': res['poster_url'],
                'plot': res['plot'],
                'imdbRating': res['rating'],
                'year': res['year']
            }
            with _cache_lock:
                _poster_cache[cache_key] = data
            return jsonify(data)
    except Exception:
        pass

    # 3. Fetch from APIs
    try:
        import requests as _req
        params = {'t': title, 'apikey': 'trilogy', 'plot': 'short'}
        if year:
            params['y'] = year
            
        for key in ['trilogy', '8a02a466', 'aa9290ec', 'f200ae9e']:
            params['apikey'] = key
            try:
                r = _req.get('https://www.omdbapi.com/', params=params, timeout=1.2)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('Response') == 'True':
                        res = {
                            'success': True, 'poster': data.get('Poster', 'N/A'),
                            'plot': data.get('Plot', ''), 'imdbRating': data.get('imdbRating', ''),
                            'year': data.get('Year', '')
                        }
                        # Save to RAM and Persistent DB
                        with _cache_lock:
                            _poster_cache[cache_key] = res
                        try:
                            conn = get_conn()
                            conn.execute('INSERT OR REPLACE INTO poster_cache (key, poster_url, plot, rating, year, success) VALUES (?,?,?,?,?,?)',
                                         (cache_key, res['poster'], res['plot'], res['imdbRating'], res['year'], 1))
                            conn.commit(); conn.close()
                        except: pass
                        return jsonify(res)
            except Exception:
                continue
        
        fail_res = {'success': False, 'poster': None, 'plot': ''}
        with _cache_lock:
            _poster_cache[cache_key] = fail_res
        return jsonify(fail_res)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Anime Poster Proxy (server-side Jikan/Kitsu fetch → avoids CORS) ─────────
@app.route('/api/anime-poster', methods=['GET'])
def anime_poster_proxy():
    """Fetch anime poster from Jikan v4 (MyAnimeList) server-side with persistent cache."""
    title = request.args.get('title', '').strip()
    if not title:
        return jsonify({'success': False, 'poster': None}), 400
        
    cache_key = f"anime_{title.lower()}"
    with _cache_lock:
        if cache_key in _anime_cache:
            return jsonify(_anime_cache[cache_key])

    # Persistent check
    try:
        conn = get_conn()
        res = conn.execute('SELECT * FROM poster_cache WHERE key = ?', (cache_key,)).fetchone()
        conn.close()
        if res:
            data = {'success': bool(res['success']), 'poster': res['poster_url'], 'source': res['source']}
            with _cache_lock:
                _anime_cache[cache_key] = data
            return jsonify(data)
    except: pass

    try:
        import requests as _req
        # ── Try Jikan v4 ────────────────────────────────────────────────
        try:
            r = _req.get('https://api.jikan.moe/v4/anime', params={'q': title, 'limit': 1}, timeout=1.5)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if data:
                    img = (data[0].get('images', {}).get('jpg', {}).get('large_image_url') or data[0].get('images', {}).get('jpg', {}).get('image_url'))
                    if img:
                        res = {'success': True, 'poster': img, 'source': 'jikan'}
                        with _cache_lock: _anime_cache[cache_key] = res
                        try:
                            conn = get_conn()
                            conn.execute('INSERT OR REPLACE INTO poster_cache (key, poster_url, source, success) VALUES (?,?,?,?)', (cache_key, img, 'jikan', 1))
                            conn.commit(); conn.close()
                        except: pass
                        return jsonify(res)
        except: pass

        # ── Fallback: Kitsu.io ──────────────────────────────────────────
        try:
            r2 = _req.get('https://kitsu.io/api/edge/anime', params={'filter[text]': title, 'page[limit]': 1}, 
                          headers={'Accept': 'application/vnd.api+json'}, timeout=1.5)
            if r2.status_code == 200:
                data2 = r2.json().get('data', [])
                if data2:
                    attrs = data2[0].get('attributes', {})
                    img2 = (attrs.get('posterImage', {}).get('large') or attrs.get('posterImage', {}).get('medium'))
                    if img2:
                        res = {'success': True, 'poster': img2, 'source': 'kitsu'}
                        with _cache_lock: _anime_cache[cache_key] = res
                        try:
                            conn = get_conn()
                            conn.execute('INSERT OR REPLACE INTO poster_cache (key, poster_url, source, success) VALUES (?,?,?,?)', (cache_key, img2, 'kitsu', 1))
                            conn.commit(); conn.close()
                        except: pass
                        return jsonify(res)
        except: pass

        fail_res = {'success': False, 'poster': None}
        with _cache_lock: _anime_cache[cache_key] = fail_res
        return jsonify(fail_res)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Helper ────────────────────────────────────────────────────────────────
def _save_rec_history(request, data, movies):
    token=request.headers.get('Authorization','').replace('Bearer ','')
    uid=get_user_from_token(token)
    if uid and movies:
        conn=get_conn()
        conn.execute('INSERT INTO rec_history (user_id,mood,genres,movie_ids) VALUES (?,?,?,?)',
                     (uid,data.get('mood',''),json.dumps(data.get('genres',[])),
                      json.dumps([m['id'] for m in movies])))
        conn.commit(); conn.close()


# ── Entry ─────────────────────────────────────────────────────────────────
if __name__=='__main__':
    init_db()

    missing = get_missing_files()
    if missing:
        print(f"\n❌ BACKEND WARNING: Missing required files: {', '.join(missing)}")
        print("   -> Bypassing background model loading to prevent crashes. Please check the React UI for instructions.\n")
    else:
        # ── WARM UP ALL ───────────────────────────────────────────────────
        from recommend import warm_up_all
        threading.Thread(target=warm_up_all, daemon=True).start()
        
        # Pre-warm sentiment model in background (eliminates first-request timeout)
        print('Pre-loading sentiment model in background...')
        threading.Thread(target=load_sentiment_model, daemon=True).start()

        # Pre-warm genre model in background (eliminates 3-5s first-request delay)
        print('Pre-loading genre model in background...')
        threading.Thread(target=load_genre_model, daemon=True).start()

        # Pre-aggregate analytics data in background (eliminates dashboard timeout)
        print('Pre-loading analytics cache in background...')
        threading.Thread(target=_build_analytics_cache, daemon=True).start()

    print('\nMoodVerse backend running at http://localhost:5000\n')
    
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()
    
    app.run(debug=True, port=5000, use_reloader=False)
