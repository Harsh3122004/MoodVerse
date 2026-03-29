import os
import shutil
import sys
import zipfile

try:
    import gdown
except ImportError:
    print("Installing required package 'gdown'...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown

def setup_datasets():
    # New File ID for the consolidated moodverse_data.zip
    file_id = '1i4liNGcDJs0fLjNXL6SnT2-9k20DrXQs'
    url = f'https://drive.google.com/uc?id={file_id}'
    zip_path = 'moodverse_data.zip'
    temp_dir = 'temp_dataset_extraction'
    
    print("\n" + "="*55)
    print(" 🚀 MoodVerse Automated Dataset & Model Downloader")
    print("="*55)
    print("\nConnecting to your public Google Drive link...")
    print("This will download the required resources (Total ~250MB).")
    print("Depending on your internet speed, this may take 1-5 minutes.")
    print("Please do not close this window.\n")

    try:
        # Download the ZIP file
        gdown.download(url, output=zip_path, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"\n❌ Download Error: {e}")
        print("Please check your internet connection or try manually downloading from the README link.")
        return

    print("\n✅ Download complete! Extracting archive...\n")
    
    try:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        print("✅ Extraction complete! Moving files to their correct directories...\n")
    except Exception as e:
        print(f"❌ Extraction Error: {e}")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    raw_dir = os.path.join(base_dir, 'datasets', 'raw')
    processed_dir = os.path.join(base_dir, 'datasets', 'processed')
    
    # Ensure local directory structure exists
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Some ZIP compressions might nest inside a folder (or not)
    # We check if the expected items are directly in temp_dir or one level deep
    source_dir = temp_dir
    items = os.listdir(temp_dir)
    if len(items) == 1 and os.path.isdir(os.path.join(temp_dir, items[0])):
        # ZIP was created by selecting the "moodverse_datasets" folder itself
        source_dir = os.path.join(temp_dir, items[0])
    
    print(f"Routing files from {source_dir}...")

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

    print("\nCleaning up temporary files...")
    try:
        shutil.rmtree(temp_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception as e:
        print(f"Warning: Cleanup Error: {e}")

    print("\n✅ Your MoodVerse environment is now fully configured!")
    print("You can now run 'python app.py' to launch the website.\n")

if __name__ == '__main__':
    setup_datasets()
