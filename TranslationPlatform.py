"""
Project 2: Enhanced Content Processing and Translation Systems:
Integrating Multilingual Resources and Real-time Language Trends

Team members:
- Jihyun Seo (Jenny)
- Taiyu Shi (Rebecca)
- Yunhang Bao
- Yuxuan Du

Features:
1. Translation Comparison (OpenAI / DeepL / Googletrans) 
2. Wikipedia Language Introduction
3. Common Voice Dataset Download (English page + dropdown menu to select language and version)
4. Extract and display five audio clips with text (randomly selected)
"""

import os
import time
import tarfile
import random
import tempfile
import requests
import streamlit as st
import openai
from bs4 import BeautifulSoup

class DownloadManager:
    """
    Class to manage the download process with better organization and error handling.
    """
    def __init__(self, email_address, lang_code, log_func):
        """
        Initialize the download manager.
        
        Args:
            email_address (str): Email address for registration
            lang_code (str): Language code for the dataset
            log_func (callable): Function to log messages
        """
        self.email = email_address
        self.lang_code = lang_code
        self.log_func = log_func
        self.url = "https://commonvoice.mozilla.org/en/datasets"
        self.download_dir = "downloads"
        self.local_filename = f"cv-corpus-{lang_code}.tar.gz"
        
        # Ensure download directory exists
        ensure_dir(self.download_dir)
        
    def setup_browser(self):
        """
        Set up and return a configured browser instance.
        
        Returns:
            tuple: (driver, wait) - Browser driver and WebDriverWait
        """
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--headless")  # Run in headless mode
        
        # Additional performance options
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        wait = WebDriverWait(driver, 20)
        
        return driver, wait
        
    def accept_cookies(self, driver, wait):
        """
        Accept cookies on the page if necessary.
        
        Args:
            driver: Browser driver
            wait: WebDriverWait instance
        """
        try:
            accept_btn = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]")),
                message="Accept button not found"
            )
            accept_btn.click()
            time.sleep(1)
        except:
            pass  # Silently continue if no cookies dialog
            
    def fill_form(self, driver, wait):
        """
        Fill the download form with email and check required boxes.
        
        Args:
            driver: Browser driver
            wait: WebDriverWait instance
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Enter email
            email_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email']")))
            email_input.clear()
            email_input.send_keys(self.email)
            self.log_func(f"Email entered: {self.email}")
            
            # Check checkboxes
            checkboxes = driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
            for i in range(min(2, len(checkboxes))):
                if not checkboxes[i].is_selected():
                    checkboxes[i].click()
                    time.sleep(0.5)
            self.log_func("Agreement checkboxes selected")
            
            return True
        except Exception as e:
            self.log_func(f"Form fill error: {str(e)[:50]}...")
            return False
            
    def select_language(self, driver, wait):
        """
        Select the target language from the dropdown.
        
        Args:
            driver: Browser driver
            wait: WebDriverWait instance
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Find language dropdown
            lang_selector = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "select[name='bundleLocale']")))
            lang_select = Select(lang_selector)
            
            # Select target language
            lang_select.select_by_value(self.lang_code)
            time.sleep(1.5)
            self.log_func(f"Language selected: {self.lang_code}")
            return True
        except Exception as e:
            self.log_func(f"Language selection error: {str(e)[:50]}...")
            return False
            
    def download_dataset(self, driver, wait):
        """
        Download the dataset using the optimal method available.
        
        Args:
            driver: Browser driver
            wait: WebDriverWait instance
            
        Returns:
            bool: True if download initiated, False otherwise
        """
        # First try direct download link
        try:
            download_link = wait.until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'Download Dataset Bundle')]")),
                message="'Download Dataset Bundle' link not found"
            )
            signed_url = download_link.get_attribute("href")
            
            if signed_url:
                self.log_func(f"Starting download to: {self.local_filename}")
                self._download_file(signed_url)
                return True
        except Exception as e:
            self.log_func("Direct download link not available")
        
        # Try download button if link not available
        try:
            download_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Download')]")),
                message="'Download' button not found"
            )
            download_button.click()
            time.sleep(5)
            self.log_func("Download initiated via button click")
            return True
        except Exception as e:
            self.log_func("Could not initiate download via button")
            return False
            
    def _download_file(self, url):
        """
        Download a file from URL with progress reporting.
        
        Args:
            url (str): URL to download from
        """
        try:
            # Use stream=True for large files
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                self.log_func(f"File size: {total_size/1024/1024:.1f} MB")
                
                # Save to file with progress reporting
                file_path = os.path.join(self.download_dir, self.local_filename)
                with open(file_path, 'wb') as f:
                    downloaded = 0
                    # Use larger chunk size for efficiency
                    for chunk in r.iter_content(chunk_size=8192*4):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Report progress at 25%, 50%, 75% and 100%
                            progress = downloaded / total_size * 100
                            if progress > 25 and downloaded % (total_size // 4) < 8192*4:
                                self.log_func(f"Download progress: {progress:.0f}%")
                
                # Create a symlink in the current directory for easier access
                if os.path.exists(self.local_filename):
                    os.remove(self.local_filename)
                os.symlink(file_path, self.local_filename)
                
                self.log_func(f"✅ Download complete: {self.local_filename}")
        except Exception as e:
            self.log_func(f"Download error: {e}")
    
    def run(self):
        """
        Run the complete download process.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.log_func(f"Starting download process for language: {self.lang_code}")
        
        driver, wait = self.setup_browser()
        success = False
        
        try:
            # Open page
            driver.get(self.url)
            self.log_func("Accessing Common Voice dataset page...")
            
            # Accept cookies
            self.accept_cookies(driver, wait)
            
            # Fill form
            if not self.fill_form(driver, wait):
                self.log_func("Could not complete form. Aborting.")
                return False
                
            # Select language
            if not self.select_language(driver, wait):
                self.log_func("Could not select language. Continuing anyway...")
            
            # Download
            success = self.download_dataset(driver, wait)
            
        except Exception as e:
            self.log_func(f"Download process error: {e}")
            success = False
        finally:
            driver.quit()
            self.log_func("Download process completed")
            
        return success# ======================
# Wikipedia Language Introduction
# ======================
LANG_WIKI_MAP = {
    "zh-CN": "https://en.wikipedia.org/wiki/Chinese_language",
    "en": "https://en.wikipedia.org/wiki/English_language",
    "ko": "https://en.wikipedia.org/wiki/Korean_language",
    "fr": "https://en.wikipedia.org/wiki/French_language",
    "de": "https://en.wikipedia.org/wiki/German_language",
    "es": "https://en.wikipedia.org/wiki/Spanish_language",
}



# Set page config at the very beginning, before any other st commands
st.set_page_config(layout="wide")

# ======================
# Sidebar: API Keys and Download Email
# ======================
st.sidebar.header("API Keys and Download Email Configuration")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
deepl_key = st.sidebar.text_input("DeepL API Key", type="password")
cv_email = st.sidebar.text_input("Common Voice Download Email")

if openai_key:
    openai.api_key = openai_key

# ======================
# Translation Functions: OpenAI, DeepL, Googletrans
# ======================
try:
    from googletrans import Translator
except ImportError:
    st.error("Please install the googletrans module: pip install googletrans==4.0.0-rc1")
    st.stop()

try:
    from deep_translator import DeeplTranslator
except ImportError:
    st.error("Please install the deep_translator module: pip install deep_translator")
    st.stop()

# Language mapping for translation
openai_lang_map = {
    "zh-CN": "Chinese (Simplified)",
    "en": "English",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}
deepl_lang_map = {
    "zh-CN": "zh",
    "en": "en",
    "ko": "ko",
    "fr": "fr",
    "de": "de",
    "es": "es",
}
googletrans_lang_map = {
    "zh-CN": "zh-cn",
    "en": "en",
    "ko": "ko",
    "fr": "fr",
    "de": "de",
    "es": "es",
}

def translate_openai(text, lang_code):
    """
    Use OpenAI GPT-3.5-turbo for translation (source language fixed to English).
    
    Args:
        text (str): Text to translate
        lang_code (str): Target language code
        
    Returns:
        str: Translated text or error message
    """
    if not openai.api_key:
        return "Please enter a valid OpenAI API Key in the sidebar"
        
    if lang_code not in openai_lang_map:
        return f"OpenAI does not support this target language: {lang_code}"
        
    target_lang_name = openai_lang_map[lang_code]
    prompt = f"Translate the following English text to {target_lang_name}:\n\n{text}\n"
    
    try:
        # Add retry mechanism
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return response['choices'][0]['message']['content'].strip()
            except (openai.error.APIError, openai.error.ServiceUnavailableError) as e:
                # Retry on service errors with backoff
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise e
    except Exception as e:
        return f"OpenAI translation error: {e}"

def translate_deepl(text, lang_code, api_key):
    """
    Use DeeplTranslator for translation (source language fixed='en').
    
    Args:
        text (str): Text to translate
        lang_code (str): Target language code
        api_key (str): DeepL API key
        
    Returns:
        str: Translated text or error message
    """
    if not api_key:
        return "Please enter a DeepL API Key in the sidebar"
        
    if lang_code not in deepl_lang_map:
        return f"DeepL does not support this target language: {lang_code}"
        
    target_lang = deepl_lang_map[lang_code]
    
    try:
        # Add caching for efficiency if the same text is translated multiple times
        cache_key = f"deepl_{text}_{target_lang}"
        if cache_key in cache:
            return cache[cache_key][0]
            
        translator = DeeplTranslator(api_key=api_key, source="en", target=target_lang)
        result = translator.translate(text)
        
        # Cache the result
        cache[cache_key] = (result, time.time())
        
        return result
    except Exception as e:
        return f"DeepL translation error: {e}"

def translate_google(text, lang_code):
    """
    Use Googletrans for translation (source language='en').
    
    Args:
        text (str): Text to translate
        lang_code (str): Target language code
        
    Returns:
        str: Translated text or error message
    """
    if lang_code not in googletrans_lang_map:
        return f"Googletrans does not support this target language: {lang_code}"
        
    target_lang = googletrans_lang_map[lang_code]
    
    # Add caching for efficiency
    cache_key = f"google_{text}_{target_lang}"
    if cache_key in cache:
        return cache[cache_key][0]
    
    try:
        translator = Translator()
        result = translator.translate(text, src="en", dest=target_lang).text
        
        # Cache the result
        cache[cache_key] = (result, time.time())
        
        return result
    except Exception as e:
        error_msg = f"Googletrans translation error: {e}"
        
        # Provide more helpful error messages
        if "429" in str(e):
            error_msg += " (Rate limit exceeded. Please try again later.)"
        elif "Connection" in str(e):
            error_msg += " (Network error. Please check your internet connection.)"
            
        return error_msg

# ======================
# Caching and Optimization Functions
# ======================
import functools
import hashlib

# Cache for expensive operations
cache = {}

def cached_operation(max_age_seconds=3600):
    """
    Decorator to cache results of expensive operations.
    
    Args:
        max_age_seconds (int): Maximum age of cached results in seconds
        
    Returns:
        Function decorator that implements caching
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = hashlib.md5(str(key_parts).encode()).hexdigest()
            
            # Check if result is in cache and not expired
            current_time = time.time()
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if current_time - timestamp < max_age_seconds:
                    return result
            
            # Execute the function and cache the result
            result = func(*args, **kwargs)
            cache[cache_key] = (result, current_time)
            return result
        return wrapper
    return decorator

@cached_operation(max_age_seconds=3600)
def scrape_language_wiki(lang_code, num_paras=3):
    """
    Cached function to scrape Wikipedia for language information.
    
    Args:
        lang_code (str): Language code (e.g., 'en', 'fr')
        num_paras (int): Number of paragraphs to retrieve
        
    Returns:
        str: Text description of the language from Wikipedia
    """
    url = LANG_WIKI_MAP.get(lang_code, "")
    if not url:
        return f"No Wikipedia page link found for {lang_code}."
    
    try:
        # Use a session for better performance
        session = requests.Session()
        resp = session.get(url, timeout=10)  # Add timeout for safety
        resp.raise_for_status()
    except Exception as e:
        return f"Error requesting {url}: {e}"
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = soup.select("p")
    
    # Filter and collect paragraphs
    results = []
    for p in paragraphs:
        txt = p.get_text().strip()
        if txt and len(txt) > 50:  # Only include substantial paragraphs
            results.append(txt)
        if len(results) >= num_paras:
            break
            
    if not results:
        return "No paragraph text could be parsed from the Wikipedia page."
        
    return "\n\n".join(results)

def ensure_dir(directory):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory (str): Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def cleanup_temp_files(temp_dir_prefix="cv_audio_", max_age_hours=24):
    """
    Clean up old temporary files to prevent disk space issues.
    
    Args:
        temp_dir_prefix (str): Prefix of temporary directories to clean
        max_age_hours (int): Maximum age of files to keep in hours
    """
    temp_dir = tempfile.gettempdir()
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    try:
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            
            # Check if it's a directory with our prefix
            if os.path.isdir(item_path) and item.startswith(temp_dir_prefix):
                # Check creation time
                stat_info = os.stat(item_path)
                age = current_time - stat_info.st_mtime
                
                if age > max_age_seconds:
                    try:
                        for subitem in os.listdir(item_path):
                            subitem_path = os.path.join(item_path, subitem)
                            os.remove(subitem_path)
                        os.rmdir(item_path)
                    except Exception as e:
                        print(f"Failed to remove {item_path}: {e}")
    except Exception as e:
        print(f"Error during temp file cleanup: {e}")

# ======================
# Common Voice Dataset Download: English page + dropdown menu
# ======================
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

def download_large_file(url, local_filename, log_func):
    """
    Download large files in chunks using requests.
    If HTML is returned, warn that download may have failed.
    """
    log_func(f"Starting download: {url}")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            ctype = r.headers.get("content-type", "").lower()
            if "text/html" in ctype:
                log_func("Warning: Server returned HTML content, possibly an error page or login required.")
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        log_func(f"Download complete: {local_filename}")
    except Exception as e:
        log_func(f"Error during download: {e}")

def scrape_common_voice_download(email_address, lang_code, log_func):
    """
    Access Common Voice dataset page and download the selected language dataset.
    Uses the DownloadManager class for better organization.
    
    Args:
        email_address (str): Email for registration
        lang_code (str): Language code to download
        log_func (callable): Function to log messages
    """
    # Create and run download manager
    manager = DownloadManager(email_address, lang_code, log_func)
    return manager.run()

# ======================
# Extract and display audio and text
# ======================
@cached_operation(max_age_seconds=3600)
def list_tar_contents(file_path, max_files=5):
    """
    List files in tar archive with caching for performance.
    
    Args:
        file_path (str): Path to tar archive
        max_files (int): Maximum number of files to list
        
    Returns:
        list or str: List of filenames or error message
    """
    if not os.path.isfile(file_path):
        return f"File {file_path} does not exist."
        
    try:
        # Try to open as gzip first
        with tarfile.open(file_path, "r:gz") as tar:
            members = sorted(tar.getmembers(), key=lambda x: x.name)
            file_list = [m.name for m in members[:max_files]]
            return file_list
    except tarfile.ReadError:
        # If not gzip, try as generic tar
        try:
            with tarfile.open(file_path, "r:*") as tar:
                members = sorted(tar.getmembers(), key=lambda x: x.name)
                file_list = [m.name for m in members[:max_files]]
                return file_list
        except Exception as e:
            return f"Failed to list files: {e}"
    except Exception as e:
        return f"Failed to list files: {e}"

def preview_corpus_samples(tar_path, sample_count=5):
    """Open tar.gz, randomly select sample_count mp3 files and display them.
    Improved to better find matching text for audio files."""
    if not os.path.isfile(tar_path):
        return None, f"File {tar_path} does not exist."
    try:
        tar = tarfile.open(tar_path, "r:gz")
    except tarfile.ReadError:
        try:
            tar = tarfile.open(tar_path, "r:*")
        except tarfile.ReadError:
            return None, "File is neither gzip compressed nor any other recognizable tar format."
        except Exception as e:
            return None, f"Failed to open archive: {e}"
    except Exception as e:
        return None, f"Failed to open archive: {e}"

    with tar:
        # Look for all possible TSV files that might contain text
        tsv_members = []
        for m in tar.getmembers():
            if m.name.endswith(".tsv"):
                tsv_members.append(m)
                
        # Sort by priority (validated.tsv should be first if found)
        tsv_members.sort(key=lambda x: 0 if "validated" in x.name.lower() else 
                                       (1 if "train" in x.name.lower() else 2))
        
        text_map = {}
        # Try to read from each TSV file until we get some mappings
        for tsv_member in tsv_members:
            try:
                f = tar.extractfile(tsv_member)
                if f:
                    lines = f.read().decode("utf-8", errors="replace").splitlines()
                    if len(lines) > 1:
                        # Try to determine the format by reading header
                        header = lines[0].split("\t")
                        
                        # Find columns for file path and text
                        file_idx = -1
                        text_idx = -1
                        
                        for i, col in enumerate(header):
                            col_lower = col.lower()
                            if "path" in col_lower or "file" in col_lower or "clip" in col_lower:
                                file_idx = i
                            elif "text" in col_lower or "sentence" in col_lower or "transcript" in col_lower:
                                text_idx = i
                        
                        # If we couldn't determine columns, make best guess
                        if file_idx == -1 or text_idx == -1:
                            if len(header) >= 3:  # Common Voice format
                                file_idx = 1  # Usually second column
                                text_idx = 2  # Usually third column
                        
                        # Parse the file
                        for line in lines[1:]:  # Skip header
                            parts = line.split("\t")
                            if len(parts) > max(file_idx, text_idx):
                                # Get filename only from path if needed
                                file_path = parts[file_idx].strip()
                                file_name = os.path.basename(file_path)
                                sentence = parts[text_idx].strip()
                                
                                # Store both with and without extension to improve matching
                                text_map[file_name] = sentence
                                if file_name.endswith(".mp3"):
                                    text_map[file_name[:-4]] = sentence
                                else:
                                    text_map[f"{file_name}.mp3"] = sentence
            except Exception as e:
                print(f"Warning: Could not process {tsv_member.name}: {e}")
        
        # Find all audio files
        mp3_members = []
        for m in tar.getmembers():
            if m.name.endswith(".mp3"):
                mp3_members.append(m)
                
        if not mp3_members:
            return None, "No mp3 files found in the archive."

        sample_count = min(sample_count, len(mp3_members))
        selected_members = random.sample(mp3_members, sample_count)

        results = []
        for m in selected_members:
            audio_path = m.name
            audio_filename = os.path.basename(audio_path)
            audio_id = os.path.splitext(audio_filename)[0]  # Remove extension
            
            tmp_dir = tempfile.mkdtemp(prefix="cv_temp_")
            extract_path = os.path.join(tmp_dir, audio_filename)
            
            try:
                data_f = tar.extractfile(m)
                if data_f:
                    with open(extract_path, "wb") as f_out:
                        f_out.write(data_f.read())
                else:
                    continue
            except Exception as e:
                return None, f"Failed to extract mp3 file: {e}"

            # Try multiple variations to find matching text
            text = ""
            for key in [audio_filename, audio_id, audio_path]:
                if key in text_map:
                    text = text_map[key]
                    break
                    
            results.append((extract_path, text))

        return results, None

# ======================
# Streamlit Main Function
# ======================
def main():
    st.title("Enhanced Content Processing and Translation Systems")
    st.markdown("""
    **Features**:
    1. Translation Comparison (OpenAI / DeepL / Googletrans)
    2. Wikipedia Language Introduction
    3. Common Voice Dataset Download
    4. Extract and display five audio clips
    """)

    # State management to persist data between interactions
    if 'translation_results' not in st.session_state:
        st.session_state.translation_results = None
    
    if 'download_logs' not in st.session_state:
        st.session_state.download_logs = []
    
    if 'audio_samples' not in st.session_state:
        st.session_state.audio_samples = None
    
    if 'wiki_data' not in st.session_state:
        st.session_state.wiki_data = None
    
    if 'download_path' not in st.session_state:
        st.session_state.download_path = None

    # Target language selection
    lang_options = [
        ("Chinese (Simplified)", "zh-CN"),
        ("English", "en"),
        ("Korean", "ko"),
        ("French", "fr"),
        ("German", "de"),
        ("Spanish", "es"),
    ]
    target_lang = st.selectbox("Select Target Language (also used for Common Voice dataset download)", lang_options, format_func=lambda x: x[0])
    target_lang_code = target_lang[1]
    
    # Create tabs for better organization
    tab1, tab4, tab2, tab3 = st.tabs(["Translation", "Language Info", "Dataset Download", "Audio Preview"])
    
    # Tab 1: Translation comparison
    with tab1:
        st.header("Translation Comparison (Input English Text)")
        text = st.text_area("Enter English text:", height=150)
        if st.button("Start Translation Comparison"):
            if text.strip():
                with st.spinner("Translating..."):
                    openai_res = translate_openai(text, target_lang_code)
                    deepl_res = translate_deepl(text, target_lang_code, deepl_key)
                    google_res = translate_google(text, target_lang_code)
                    
                # Store results in session state to persist between interactions
                st.session_state.translation_results = {
                    "text": text,
                    "openai": openai_res,
                    "deepl": deepl_res,
                    "google": google_res,
                    "lang_code": target_lang_code
                }
            else:
                st.warning("Please enter English text before translating.")
        
        # Display translation results if available
        if st.session_state.translation_results:
            results = st.session_state.translation_results
            
            # Only show results if they match the current language
            if results["lang_code"] == target_lang_code:
                cols = st.columns(3)
                with cols[0]:
                    st.subheader("OpenAI")
                    st.text_area("", results["openai"], height=200)
                with cols[1]:
                    st.subheader("DeepL")
                    st.text_area("", results["deepl"], height=200)
                with cols[2]:
                    st.subheader("Googletrans")
                    st.text_area("", results["google"], height=200)
            else:
                st.info(f"Translation results for {results['lang_code']} are available but don't match current language selection.")
                
    # Tab 2: Wikipedia language info
    with tab4:
        st.header("Wikipedia Language Introduction")
        
        # Get data if not already in session state or if language changed
        if (st.session_state.wiki_data is None or 
            'lang_code' not in st.session_state.wiki_data or 
            st.session_state.wiki_data['lang_code'] != target_lang_code):
            
            with st.spinner("Fetching language information..."):
                wiki_text = scrape_language_wiki(target_lang_code, num_paras=3)
                st.session_state.wiki_data = {
                    'lang_code': target_lang_code,
                    'text': wiki_text
                }
        
        # Display the Wikipedia information - ONLY ONCE
        if st.session_state.wiki_data and 'text' in st.session_state.wiki_data:
            st.markdown(st.session_state.wiki_data['text'])
        else:
            st.error("Failed to retrieve language information from Wikipedia.")
            
    # Tab 3: Common Voice dataset download
    with tab2:
        st.header("Common Voice Dataset Download")
        st.markdown("Please enter your email in the sidebar, then click the button below to start downloading the dataset for the selected language.")
        
        download_col1, download_col2 = st.columns([1, 1])
        
        with download_col1:
            if st.button("Download Dataset"):
                if not cv_email.strip():
                    st.warning("Please enter your Common Voice download email in the sidebar first.")
                else:
                    # Clear previous logs only if starting a new download
                    st.session_state.download_logs = []
                    
                    def log_func(msg):
                        st.session_state.download_logs.append(msg)
                        # Use a placeholder to update logs in real-time
                        with download_status:
                            st.write(msg)
                    
                    # Create a placeholder for download status
                    download_status = st.empty()
                    with st.spinner("Downloading dataset, please wait..."):
                        success = scrape_common_voice_download(cv_email, target_lang_code, log_func)
                    
                    # Check if file exists and update UI
                    tar_name = f"cv-corpus-{target_lang_code}.tar.gz"
                    if os.path.exists(tar_name):
                        with download_status:
                            st.success(f"Download complete: {tar_name}")
                        
                        # Store download path in session state for other tabs to use
                        st.session_state.download_path = tar_name
                        
                        # Show file listing in the second column
                        with download_col2:
                            st.subheader("Archive Contents")
                            listing = list_tar_contents(tar_name, max_files=5)
                            if isinstance(listing, list):
                                st.write("\n".join(listing))
                            else:
                                st.error(listing)
                    else:
                        with download_status:
                            st.warning("Archive not found in current directory. It might be in your browser's default download folder.")
        
            # Check if a previous download exists
            tar_name = f"cv-corpus-{target_lang_code}.tar.gz"
            if os.path.exists(tar_name) and not st.session_state.download_logs:
                st.success(f"Dataset file already exists: {tar_name}")
                st.session_state.download_path = tar_name
                
        # Show download logs if available
        if st.session_state.download_logs:
            with download_col2:
                with st.expander("Download Logs", expanded=False):
                    st.text("\n".join(st.session_state.download_logs))
    
    # Tab 4: Audio preview
    with tab3:
        st.header("Audio Preview")
        tar_path = f"cv-corpus-{target_lang_code}.tar.gz"
        
        if not os.path.exists(tar_path):
            st.info(f"Dataset file for '{target_lang[0]}' not found. Please download it first in the 'Dataset Download' tab.")
        else:
            if st.button("Extract and Play 5 Random Audio Clips"):
                # Clean up old temporary files before creating new ones
                cleanup_temp_files()
                
                with st.spinner("Extracting audio samples..."):
                    audio_files, err = preview_corpus_samples(tar_path, sample_count=5)
                    
                    if audio_files is None:
                        st.error(f"Error extracting samples: {err}")
                    else:
                        # Store in session state to persist between interactions
                        st.session_state.audio_samples = audio_files
            
             # Display audio samples if available (either from current or previous extraction)
            if st.session_state.audio_samples:
                samples = st.session_state.audio_samples
                
                for idx, sample in enumerate(samples, start=1):
                    st.markdown(f"#### Audio Sample {idx}")
                    
                    # Handle both tuple format (path, text) and string format
                    if isinstance(sample, tuple):
                        audio_path, text = sample
                    else:
                        audio_path = sample
                        text = ""
                    
                    # Check if the file exists (might have been cleaned up)
                    if os.path.exists(audio_path):
                        st.audio(audio_path)
                        if text:
                            st.markdown(f"**Transcript**: {text}")
                    else:
                        st.warning("Audio file is no longer available. Please extract again.")
                    
                    st.markdown("---")
    

if __name__ == "__main__":
    main()