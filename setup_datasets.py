import os
import shutil
import sys

try:
    import gdown
except ImportError:
    print("Installing required package 'gdown'...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown

def setup_datasets():
    folder_id = '1861lOpGNh6cAe-A-5JUda5AJExZp7AsM'
    url = f'https://drive.google.com/drive/folders/{folder_id}'
    temp_dir = 'temp_dataset_downloads'
    
    print("\n" + "="*55)
    print(" 🚀 MoodVerse Automated Dataset & Model Downloader")
    print("="*55)
    print("\nConnecting to your public Google Drive URL...")
    print("This will download the required resources (Total ~250MB).")
    print("Depending on your internet speed, this may take 1-5 minutes.")
    print("Please do not close this window.\n")

    try:
        # Proceed with downloading
        # The remaining_ok=True flag helps resume downloads if it drops
        gdown.download_folder(url, output=temp_dir, quiet=False, use_cookies=False, remaining_ok=True)
    except Exception as e:
        print(f"\n❌ Download Error: {e}")
        print("Please check your internet connection or try manually downloading from the README link.")
        return

    print("\n✅ Download complete! Moving files to their correct directories...\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    raw_dir = os.path.join(base_dir, 'datasets', 'raw')
    processed_dir = os.path.join(base_dir, 'datasets', 'processed')
    
    # Ensure local directory structure exists
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Some versions of gdown nest the download inside a folder bearing the Drive folder's name
    source_dir = temp_dir
    subdirs = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f))]
    if len(subdirs) == 1 and 'raw' not in subdirs and 'processed' not in subdirs:
        source_dir = os.path.join(temp_dir, subdirs[0])
        
    print(f"Extracting from archive routing...")

    # 1. Move svd_artifacts.pkl to models/
    svd_src = os.path.join(source_dir, 'svd_artifacts.pkl')
    if os.path.exists(svd_src):
        d = os.path.join(models_dir, 'svd_artifacts.pkl')
        if os.path.exists(d): os.remove(d)
        shutil.move(svd_src, d)
        print(" -> Placed svd_artifacts.pkl in models/")

    # 2. Move raw data to datasets/raw/
    raw_src = os.path.join(source_dir, 'raw')
    if os.path.exists(raw_src) and os.path.isdir(raw_src):
        for item in os.listdir(raw_src):
            s = os.path.join(raw_src, item)
            d = os.path.join(raw_dir, item)
            if os.path.exists(d):
                if os.path.isdir(d): shutil.rmtree(d)
                else: os.remove(d)
            shutil.move(s, d)
        print(" -> Placed raw dataset files in datasets/raw/")

    # 3. Move processed data to datasets/processed/
    processed_src = os.path.join(source_dir, 'processed')
    if os.path.exists(processed_src) and os.path.isdir(processed_src):
        for item in os.listdir(processed_src):
            s = os.path.join(processed_src, item)
            d = os.path.join(processed_dir, item)
            if os.path.exists(d):
                if os.path.isdir(d): shutil.rmtree(d)
                else: os.remove(d)
            shutil.move(s, d)
        print(" -> Placed clean dataset files in datasets/processed/")

    print("\nCleaning up temporary extraction files...")
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not delete temp directory {temp_dir}: {e}")

    print("\n✅ All datasets and models are successfully configured!")
    print("You can now run 'python app.py' to launch the MoodVerse website.\n")

if __name__ == '__main__':
    setup_datasets()
