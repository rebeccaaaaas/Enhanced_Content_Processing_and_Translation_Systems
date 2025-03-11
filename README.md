# Enhanced Content Processing and Translation Systems

A multilingual platform that integrates translation services, language resources, and audio dataset management capabilities.

## Features

- **Translation Comparison**: Compare translations between OpenAI, DeepL, and Google Translate in real-time
- **Wikipedia Language Introduction**: Access concise information about selected languages
- **Common Voice Dataset Download**: Easily download language datasets from Mozilla's Common Voice project
- **Audio Preview**: Extract and play audio samples from downloaded datasets

## Demo

![Audio Preview Demo](./assets/translation.gif)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/translation-platform.git
cd translation-platform
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
   - Get an OpenAI API key from [OpenAI](https://platform.openai.com/)
   - Get a DeepL API key from [DeepL](https://www.deepl.com/pro-api)
   - Create an account at [Mozilla Common Voice](https://commonvoice.mozilla.org/) for dataset downloads

## Usage

1. Run the Streamlit app:
```bash
streamlit run TranslationPlatform.py
```

2. Open your browser and go to http://localhost:8501

3. Enter your API keys in the sidebar:
   - OpenAI API Key
   - DeepL API Key
   - Common Voice Email (for dataset downloads)

4. Use the interface to select languages and explore the features!

## Project Structure

```
.
├── TranslationPlatform.py    # Main application file
├── requirements.txt          # Project dependencies
├── downloads/                # Folder for downloaded datasets
├── assets/                   # Demo GIFs and images
└── README.md                 # Project documentation
```

## Technologies Used

- **Streamlit**: Web application framework
- **OpenAI API**: AI-powered language translation
- **DeepL API**: Professional translation service
- **Google Translate**: Additional translation service
- **Selenium**: Web automation for dataset downloads
- **BeautifulSoup**: Web scraping for language information
- **Requests**: HTTP requests handling
- **Python**: Core programming language


## Note on Dataset Downloads

- The Common Voice dataset download process may be slow depending on the selected language and your internet connection
- The application automatically handles cookie acceptance, form filling, and download initiation
- Downloaded datasets are cached to prevent redundant downloads

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Mozilla Common Voice](https://commonvoice.mozilla.org/) for providing open-source speech datasets
- [Wikipedia](https://www.wikipedia.org/) for language information
- OpenAI, DeepL, and Google for translation services