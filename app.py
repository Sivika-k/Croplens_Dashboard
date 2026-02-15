from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib 
import traceback 
import os 

app = Flask(__name__)

# --- GLOBAL MODEL AND DATA SETUP ---
MODEL = None
DATA_DF = None 
# Ensure this path is correct relative to app.py
MODEL_PATH = 'models/gbm_rice_predictor.joblib' 

# Global lists will hold English names after load_data() runs
rice_types = []
districts = []
district_market_map = {}


# --- Data Loading Function ---
def load_data():
    """Loads and prepares data, and attempts to load the GBM model."""
    global DATA_DF, MODEL, rice_types, districts, district_market_map

    # Ensure the data file exists
    data_file = 'data/rice_data.csv'
    if not os.path.exists(data_file):
        print(f"FATAL ERROR: Data file not found at {data_file}. Using generic dummy data.")
        data = {'date': ['2025-01-01'], 'variety': ['Basmathi'], 'district': ['Chennai'], 'market': ['Chennai Market'], 'modal_price': [5000]}
        DATA_DF = pd.DataFrame(data)
        DATA_DF['date'] = pd.to_datetime(DATA_DF['date'])
        rice_types = ['Basmathi']
        districts = ['Chennai'] # Correct list assignment
        district_market_map = {'Chennai': ['Chennai Market']}
        return 

    try:
        # 1. Load Data
        df = pd.read_csv(data_file)
        # Corrected column mapping: District must map to 'district'
        df.rename(columns={
            'Date': 'date', 
            'Rice Variety': 'variety', 
            'District': 'district', 
            'Market Name': 'market', 
            'Rice Price (₹/Quintal)': 'modal_price'  
        }, inplace=True)
        df['date'] = pd.to_datetime(df['date']) 
        DATA_DF = df 
        
        # 2. Prepare UI Lists
        rice_types = sorted(DATA_DF['variety'].unique().tolist())
        districts = sorted(DATA_DF['district'].unique().tolist())
        district_market_map = DATA_DF.groupby('district')['market'].unique().apply(list).to_dict()

    except Exception as e:
        print(f"FATAL ERROR during data loading: {e}. Using generic dummy data.")
        traceback.print_exc()
        # Fallback dictionary and list creation
        data = {'date': ['2025-01-01'], 'variety': ['Basmathi'], 'district': ['Chennai'], 'market': ['Chennai Market'], 'modal_price': [5000]}
        DATA_DF = pd.DataFrame(data)
        DATA_DF['date'] = pd.to_datetime(DATA_DF['date'])
        rice_types = ['Basmathi']
        districts = ['Chennai']
        district_market_map = {'Chennai': ['Chennai Market']}


    # 3. Load Model
    try:
        if os.path.exists(MODEL_PATH):
            MODEL = joblib.load(MODEL_PATH)
            print(f"✅ Gradient Boosting Model loaded successfully from {MODEL_PATH}.")
        else:
            print(f"⚠️ WARNING: Model file not found at {MODEL_PATH}. Prediction will use DUMMY LOGIC.")
    except Exception as e:
        print(f"❌ ERROR loading GBM model: {e}") 
        traceback.print_exc()

load_data()


# --- PREDICTION LOGIC FUNCTION ---
def get_prediction(rice_type, district, market, start_date_str, end_date_str):
    """Generates the prediction using the loaded model or dummy data."""
    
    global MODEL 
    
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    prediction_duration = (end_date - start_date).days + 1
    dates_list = pd.date_range(start=start_date, periods=prediction_duration, freq='D')
    
    
    if MODEL is not None:
        try:
            # --- REAL MODEL PREDICTION PATH ---
            future_df = pd.DataFrame({
                'date': dates_list, 
                'variety': [rice_type] * prediction_duration, 
                'district': [district] * prediction_duration, 
                'market': [market] * prediction_duration,
            })
            
            # --- Model Feature Engineering ---
            future_df['Year'] = future_df['date'].dt.year
            future_df['Month'] = future_df['date'].dt.month
            future_df['Day'] = future_df['date'].dt.day
            
            all_varieties = DATA_DF['variety'].unique()
            all_districts = DATA_DF['district'].unique()
            all_markets = DATA_DF['market'].unique() 
            
            input_data = future_df[['Year', 'Month', 'Day']].copy()
            
            # One-Hot Encoding
            for var in sorted(all_varieties):
                input_data[f'variety_{var}'] = (future_df['variety'] == var).astype(int)
            for dist in sorted(all_districts):
                input_data[f'district_{dist}'] = (future_df['district'] == dist).astype(int)
            for market_name in sorted(all_markets):
                input_data[f'market_{market_name}'] = (future_df['market'] == market_name).astype(int)
            
            # Prepare final input data, ensuring column order matches the model
            # This is a robust way to handle feature order mismatch
            model_features = sorted(MODEL.feature_names_in_) if hasattr(MODEL, 'feature_names_in_') else sorted(input_data.columns)
            input_data_final = input_data.reindex(columns=model_features, fill_value=0)
            
            predictions = MODEL.predict(input_data_final)
            predictions = np.maximum(4000, predictions) # Cap minimum price
            prices_list = predictions.round(0).astype(int).tolist() 

        except Exception as e:
            print(f"Prediction Error in Model Path. Falling back to dummy logic: {e}")
            traceback.print_exc()
            MODEL = None 
            # Fallback to dummy
            filtered_df = DATA_DF[(DATA_DF['variety'] == rice_type) & (DATA_DF['district'] == district) & (DATA_DF['market'] == market)].sort_values(by='date')
            last_known_price = filtered_df['modal_price'].iloc[-1] if not filtered_df.empty else 4500
            
            base_prices = last_known_price + np.arange(prediction_duration) * 5 
            random_fluctuation = np.random.randint(-20, 20, size=prediction_duration)
            prices_list = (base_prices + random_fluctuation).round(0).astype(int).tolist()
            
    else:
        # --- DUMMY PREDICTION PATH ---
        filtered_df = DATA_DF[(DATA_DF['variety'] == rice_type) & (DATA_DF['district'] == district) & (DATA_DF['market'] == market)].sort_values(by='date')
        last_known_price = filtered_df['modal_price'].iloc[-1] if not filtered_df.empty else 4500
        
        base_prices = last_known_price + np.arange(prediction_duration) * 5 
        random_fluctuation = np.random.randint(-20, 20, size=prediction_duration)
        prices_list = (base_prices + random_fluctuation).round(0).astype(int).tolist()
        
    return dates_list.strftime('%Y-%m-%d').tolist(), prices_list


# --- TRANSLATION DATA (COMPLETE TAMIL CONTENT) ---
TAMIL_CONTENT = {
    # --- Index Page Content ---
    'page_title': 'க்ராப்லென்ஸ் | அரிசி விலை முன்னறிவிப்பு',
    'main_heading': 'அரிசி விலை தகவல் அமைப்பு',
    'type_of_rice': 'அரிசியின் வகை',
    'select_placeholder': 'தேர்ந்தெடு',
    'district_label': 'மாவட்டம்',
    'select_district': 'மாவட்டத்தை தேர்ந்தெடுங்கள்',
    'market_label': 'சந்தை',
    'select_market_first': 'முதலில் மாவட்டத்தைத் தேர்ந்தெடுக்கவும்',
    'date_range': 'தேதி வரம்பு',
    'go_button': 'போகவும்',
    'home_tab': 'முகப்பு',
    'about_us_tab': 'எங்களைப் பற்றி',
    'privacy_policy_tab': 'தனியுரிமைக் கொள்கை',
    'mobile_app_tab': 'மொபைல் செயலி',
    'reach_us_tab': 'எங்களைத் தொடர்புகொள்ள',
    'copyright': 'அனைத்து உரிமைகளும் பாதுகாக்கப்பட்டவை',
    'support': 'ஆதரவு',
    'select_market': 'சந்தையைத் தேர்ந்தெடுங்கள்', 
    
    # --- Mobile App Page Content ---
    'mobile_app_heading': 'க்ராப்லென்ஸ் மொபைல் செயலியை பதிவிறக்கம் செய்யுங்கள்',
    'mobile_app_subtext': 'அரிசி விலை முன்னறிவிப்புகள் மற்றும் சந்தை நுண்ணறிவுகளை உடனுக்குடன் உங்கள் பாக்கெட்டில் பெறுங்கள். சந்தைக்கு ஒரு படி மேலே இருங்கள்!',
    'section1_title': 'பயணத்தின்போது AI முன்னறிவிப்பு',
    'section1_content1': 'க்ராப்லென்ஸ் மொபைல் செயலி எங்கள் கிராடியன்ட் பூஸ்டிங் மெஷின் (GBM) மாதிரியின் ஆற்றலை உங்கள் விரல் நுனியில் கொண்டுவருகிறது. உங்கள் குறிப்பிட்ட அரிசி வகை மற்றும் உள்ளூர் சந்தைக்கான 30 நாட்கள் முன்னோக்கு விலை முன்னறிவிப்புகளைப் பெறுங்கள், சரியான நேரத்தில் மற்றும் லாபகரமான விற்பனை முடிவுகளை நீங்கள் எடுக்கிறீர்கள் என்பதை உறுதிசெய்கிறது.',
    'section1_feature1': 'உடனடி 30 நாள் விலை போக்குகள்.',
    'section1_feature2': 'தமிழ்நாடு மாவட்டங்களுக்கான உள்ளூர் சந்தை தரவு.',
    'section1_feature3': 'நல்ல விற்பனை விலையை தவறவிடாமல் இருக்க விலை எச்சரிக்கைகளை அமைக்கவும்.',
    'section2_title': 'எளிமைப்படுத்தப்பட்ட பயனர் அனுபவம்',
    'section2_content1': 'விவசாயிகள் மற்றும் வர்த்தகர்களை மனதில் கொண்டு வடிவமைக்கப்பட்டுள்ளது, இந்த செயலி எளிமையான, உள்ளுணர்வு இடைமுகத்தை கொண்டுள்ளது. சிக்கலான தரவை அணுகுவது இவ்வளவு எளிமையாக இருந்ததில்லை—உங்கள் பயிர் மற்றும் சந்தையைத் தேர்ந்தெடுக்க சில தட்டுகள் மட்டுமே, முன்னறிவிப்பு தயாராக உள்ளது!',
    'section2_feature1': 'பல மொழி ஆதரவு (ஆங்கிலம் மற்றும் தமிழ்).',
    'section2_feature2': 'மின்னல் வேகத்தில் ஏற்றும் வேகம்.',
    'section2_feature3': 'ஒருங்கிணைந்த வானிலை தகவல் (வரவிருக்கும் அம்சம்).',
    'get_started_today': 'இன்றே தொடங்கவும்!',
    'download_google_play': 'Google Play-இல் பதிவிறக்கம் செய்யுங்கள்',
    'download_app_store': 'App Store-இல் பதிவிறக்கம் செய்யுங்கள்',
    'os_requirement': '*Android 5.0+ அல்லது iOS 12.0+ தேவை',
    
    # --- About Us Page Content ---
    'about_us_title': 'க்ராப்லென்ஸ் | எங்களைப் பற்றி',
    'about_us_heading': 'க்ராப்லென்ஸ் பற்றி',
    'about_us_intro': 'க்ராப்லென்ஸ் என்பது தமிழ்நாடு விவசாயிகள் மற்றும் வர்த்தகர்களுக்கு துல்லியமான, முன்னோக்கிய அரிசி சந்தை நுண்ணறிவுகளை வழங்குவதற்காக அர்ப்பணிக்கப்பட்ட ஒரு முன்முயற்சியாகும். எங்கள் தளம் மேம்பட்ட இயந்திர கற்றல் (ML) மாதிரிகளை, குறிப்பாக கிரேடியன்ட் பூஸ்டிங் மெஷின் (GBM) மாதிரியைப் பயன்படுத்துகிறது, இது வரலாற்று சந்தை தரவுகளின் அடிப்படையில் 30 நாட்கள் வரை அரிசி விலையை முன்கூட்டியே கணிக்கிறது.',
    'mission_heading': 'எங்கள் நோக்கம்',
    'mission_text': 'சந்தை அபாயங்களைக் குறைத்து, விவசாய சமூகத்திற்கான லாபத்தை அதிகரிப்பதே எங்கள் நோக்கம். வெளிப்படையான மற்றும் சரியான நேரத்தில் விலை முன்னறிவிப்புகளை வழங்குவதன் மூலம், தகவல் சமச்சீரற்ற தன்மையை நீக்கி, ஒவ்வொரு பங்குதாரரும் தகவலறிந்த முடிவுகளை எடுக்க உதவுவதை நாங்கள் நோக்கமாகக் கொண்டுள்ளோம்.',
    'technology_heading': 'கணிப்புக்கான தொழில்நுட்பம்',
    'tech_li1_title': 'தரவு உந்துதல் மாதிரி:',
    'tech_li1_text': 'தினசரி விலை ஏற்ற இறக்கங்கள், பருவ கால போக்குகள் மற்றும் உள்ளூர் சந்தை காரணிகளை பகுப்பாய்வு செய்யும் ஒரு வலுவான கிரேடியன்ட் பூஸ்டிங் மெஷின் (GBM) மாதிரியை நாங்கள் பயன்படுத்துகிறோம்.',
    'tech_li2_title': 'ஹைப்பர்-உள்ளூர் துல்லியம்:',
    'tech_li2_text': 'எங்கள் கணிப்புகள் குறிப்பிட்ட அரிசி வகைகள், மாவட்டங்கள் மற்றும் தமிழ்நாட்டிற்குள் உள்ள சந்தைகளுக்கு ஏற்ப வடிவமைக்கப்பட்டுள்ளன, இது பொருத்தத்தையும் துல்லியத்தையும் உறுதி செய்கிறது.',
    'tech_li3_title': 'தொடர்ச்சியான கற்றல்:',
    'tech_li3_text': 'மாறும் பொருளாதார மற்றும் காலநிலை நிலைமைகளுக்கு ஏற்றவாறு, சமீபத்திய சந்தை தரவுகளுடன் மாதிரி தொடர்ந்து மறுபயிற்சி செய்யப்படுகிறது.',
    'team_heading': 'எங்கள் குழு',
    'team_name1': 'டாக்டர். கே. அருள்',
    'team_role1': 'முன்னணி தரவு விஞ்ஞானி, ML மாதிரி மேம்பாடு',
    'team_name2': 'எஸ். கவிதா',
    'team_role2': 'துறை நிபுணர், விவசாய பொருளாதாரம்',
    'team_name3': 'வி. ஆனந்த்',
    'team_role3': 'மென்பொருள் கட்டிடக் கலைஞர், வலை மற்றும் செயலி மேம்பாடு',
    
    # --- Privacy Policy Page Content ---
    'privacy_policy_title': 'க்ராப்லென்ஸ் | தனியுரிமைக் கொள்கை',
    'privacy_policy_heading': 'க்ராப்லென்ஸ் தனியுரிமைக் கொள்கை',
    'privacy_policy_date': 'செயல்பாட்டு தேதி: ஜனவரி 1, 2025',
    'privacy_intro': 'எங்கள் பயனர்களின் தனியுரிமையைப் பாதுகாப்பதில் க்ராப்லென்ஸ் உறுதியாக உள்ளது. எங்கள் அரிசி விலை முன்னறிவிப்பு அமைப்பைப் பயன்படுத்தும்போது உங்கள் தகவலை நாங்கள் எவ்வாறு சேகரிக்கிறோம், பயன்படுத்துகிறோம், வெளியிடுகிறோம் மற்றும் பாதுகாக்கிறோம் என்பதை இந்தக் கொள்கை விளக்குகிறது.',
    'data_collection_heading': '1. நாங்கள் சேகரிக்கும் தகவல்',
    'data_collection_intro': 'நாங்கள் இரண்டு வகையான தகவல்களை சேகரிக்கிறோம்:',
    'data_type1_title': 'தனிப்பட்டதல்லாத தகவல்:',
    'data_type1_text': 'இது முன்னறிவிப்பிற்காக தேர்ந்தெடுக்கப்பட்ட அரிசி வகை, மாவட்டம், சந்தை மற்றும் தேதி வரம்பு போன்ற உங்கள் பயன்பாடு தொடர்பான தரவை உள்ளடக்கியது. இந்தத் தரவு அநாமதேயமாக்கப்பட்டு, எங்கள் கணிப்பு மாதிரிகள் மற்றும் சேவை தரத்தை மேம்படுத்த மட்டுமே பயன்படுத்தப்படுகிறது.',
    'data_type2_title': 'தனிப்பட்ட தகவல் (தன்னார்வமானது):',
    'data_type2_text': '"எங்களைத் தொடர்புகொள்ள" படிவம் மூலம் நீங்கள் எங்களைத் தொடர்பு கொண்டால், உங்கள் பெயர், மின்னஞ்சல் முகவரி மற்றும் செய்தியை நாங்கள் சேகரிக்கிறோம். இந்தத் தகவல் உங்கள் கேள்விக்கு பதிலளிக்க மட்டுமே பயன்படுத்தப்படுகிறது.',
    'data_usage_heading': '2. உங்கள் தகவலை நாங்கள் எவ்வாறு பயன்படுத்துகிறோம்',
    'data_usage_text': 'நாங்கள் சேகரிக்கும் தகவலை பின்வரும் நோக்கங்களுக்காகப் பயன்படுத்துகிறோம்:',
    'usage_li1': 'அரிசி விலை முன்னறிவிப்புகளை உருவாக்குவது உட்பட க்ராப்லென்ஸ் சேவையை வழங்கவும் இயக்கவும்.',
    'usage_li2': 'செயல்பாடு மற்றும் துல்லியத்தை மேம்படுத்த எங்கள் சேவைகள் தொடர்பான போக்குகள், பயன்பாடு மற்றும் செயல்பாடுகளை கண்காணிக்கவும் பகுப்பாய்வு செய்யவும்.',
    'usage_li3': 'உங்கள் கருத்துகள், கேள்விகள் மற்றும் கோரிக்கைகளுக்குப் பதிலளிக்க (தனிப்பட்ட தகவல் வழங்கப்பட்டால்).',
    'data_sharing_heading': '3. உங்கள் தகவலைப் பகிர்வது',
    'data_sharing_text': 'உங்கள் தனிப்பட்டதல்லாத அல்லது தனிப்பட்ட தகவலை மூன்றாம் தரப்பினருக்கு அவர்களின் சந்தைப்படுத்தல் நோக்கங்களுக்காக நாங்கள் பகிரவோ, விற்கவோ அல்லது வாடகைக்கு விடவோ மாட்டோம். விவசாய வளர்ச்சியை மேம்படுத்த ஆராய்ச்சிப் பங்காளிகள் அல்லது அரசாங்க அமைப்புகளுடன் அநாமதேய, திரட்டப்பட்ட பயன்பாட்டுத் தரவைப் பகிரலாம், ஆனால் இந்தத் தரவு எந்தவொரு தனிப்பட்ட பயனரையும் அடையாளம் காணாது.',
    'cookies_heading': '4. குக்கீகள் மற்றும் கண்காணிப்பு',
    'cookies_text': 'உங்கள் வருகையின் போது உங்கள் மொழி விருப்பத்தை (ஆங்கிலம்/தமிழ்) நிர்வகிக்க மட்டுமே நாங்கள் அடிப்படை அமர்வு குக்கீகளைப் பயன்படுத்துகிறோம். இந்த குக்கீகள் தற்காலிகமானவை மற்றும் இலக்கு விளம்பரம் அல்லது நிரந்தர கண்காணிப்புக்காகப் பயன்படுத்தப்படுவதில்லை.',
    'contact_heading': '5. எங்களைத் தொடர்புகொள்ளவும்',
    'contact_text': 'இந்த தனியுரிமைக் கொள்கையைப் பற்றி உங்களுக்கு கேள்விகள் அல்லது கவலைகள் இருந்தால், "எங்களைத் தொடர்புகொள்ள" பக்கம் மூலம் எங்களைத் தொடர்பு கொள்ளவும்.',
    
    # --- Reach Us Page Content ---
    'reach_us_title': 'க்ராப்லென்ஸ் | எங்களைத் தொடர்புகொள்ள',
    'reach_us_heading': 'எங்களைத் தொடர்புகொள்ளவும்',
    'reach_us_intro': 'உங்களுக்கு ஏதேனும் கேள்விகள், பின்னூட்டங்கள் அல்லது ஆதரவு தேவைப்பட்டால், கீழே உள்ள படிவத்தை நிரப்பவும் அல்லது எங்களை நேரடியாகத் தொடர்பு கொள்ளவும். உங்கள் கருத்து எங்களுக்கு முக்கியமானது.',
    'form_name': 'உங்கள் பெயர்',
    'form_email': 'மின்னஞ்சல் முகவரி',
    'form_subject': 'பொருள்',
    'form_message': 'உங்கள் செய்தி',
    'form_send_button': 'அனுப்பு',
    'contact_info_heading': 'தொடர்பு தகவல்',
    'contact_info_text': 'விவசாயம் மற்றும் சந்தை நுண்ணறிவு தொடர்பான வினவல்களுக்கு, எங்களை மின்னஞ்சல் அல்லது தொலைபேசி மூலம் அணுகலாம்.',
    'contact_address': 'முகவரி',
    'contact_address_line1': 'க்ராப்லென்ஸ் திட்டப்பணி',
    'contact_address_line2': 'கோயம்புத்தூர், தமிழ்நாடு, இந்தியா',
    'contact_phone': 'தொலைபேசி',
    'contact_email_label': 'மின்னஞ்சல்',
    'contact_phone_number': '+91 98765 43210',
    'contact_email_address': 'support@croplens.in',
}

# Tamil district list (for display purposes)
TAMIL_DISTRICTS = [
    'அரியலூர்', 'செங்கல்பட்டு', 'சென்னை', 'கோயம்புத்தூர்', 'கடலூர்', 'தருமபுரி',
    'திண்டுக்கல்', 'ஈரோடு', 'கள்ளக்குறிச்சி', 'காஞ்சிபுரம்', 'கன்னியாகுமரி', 'கரூர்', 
    'கிருஷ்ணகிரி', 'மதுரை', 'மயிலாடுதுறை', 'நாகப்பட்டினம்', 'நாமக்கல்', 'பெரம்பலூர்', 
    'புதுக்கோட்டை', 'ராமநாதபுரம்', 'இராணிப்பேட்டை', 'சேலம்', 'சிவகங்கை', 'தென்காசி', 
    'தஞ்சாவூர்', 'நீலகிரி', 'தேனி', 'திருவள்ளூர்', 'திருவாரூர்', 'தூத்துக்குடி',
    'திருச்சிராப்பள்ளி', 'திருநெல்வேலி', 'திருப்பத்தூர்', 'திருப்பூர்', 'திருவண்ணாமலை', 
    'வேலூர்', 'விழுப்புரம்', 'விருதுநகர்'
]

# --- Route Definitions ---

@app.route('/', defaults={'lang': 'en'})
@app.route('/<lang>/', methods=['GET'])
def home(lang):
    """Handles the main page load with language switching."""
    current_lang = lang if lang in ['en', 'ta'] else 'en'
    
    # Use zip for the Tamil language version to provide Tamil district names
    zipped_districts = list(zip(districts, TAMIL_DISTRICTS))

    if current_lang == 'ta':
        return render_template('index.html', 
                               rice_types=rice_types, 
                               districts_data=zipped_districts,
                               district_market_map=district_market_map,
                               content=TAMIL_CONTENT,
                               lang='ta')
    else:
        return render_template('index.html', 
                               rice_types=rice_types, 
                               districts=districts,
                               district_market_map=district_market_map,
                               content={}, 
                               lang='en')


@app.route('/croplens', methods=['POST']) 
def croplens_prediction():
    """Handles the form submission and returns the prediction."""
    rice_type = request.form.get('rice_type')
    district = request.form.get('district') 
    market = request.form.get('market')     
    date_from_str = request.form.get('date_from') 
    date_to_str = request.form.get('date_to') 
    
    dates_list, prices_list = get_prediction(rice_type, district, market, date_from_str, date_to_str)
    
    return render_template('forecast.html', 
                            rice_type=rice_type, district=district, market=market, dates=dates_list, prices=prices_list) 


@app.route('/about_us/', defaults={'lang': 'en'})
@app.route('/<lang>/about_us/', methods=['GET'])
def about_us(lang):
    """Renders the About Us page with language support."""
    current_lang = lang if lang in ['en', 'ta'] else 'en'
    
    return render_template('about_us.html', 
                           content=TAMIL_CONTENT if current_lang == 'ta' else {},
                           lang=current_lang)

@app.route('/privacy_policy/', defaults={'lang': 'en'})
@app.route('/<lang>/privacy_policy/', methods=['GET'])
def privacy_policy(lang):
    """Renders the Privacy Policy page with language support."""
    current_lang = lang if lang in ['en', 'ta'] else 'en'
    return render_template('privacy_policy.html', 
                           content=TAMIL_CONTENT if current_lang == 'ta' else {}, 
                           lang=current_lang)

@app.route('/mobile_app/', defaults={'lang': 'en'})
@app.route('/<lang>/mobile_app/', methods=['GET'])
def mobile_app(lang):
    """Renders the Mobile App information page with language support."""
    current_lang = lang if lang in ['en', 'ta'] else 'en'
    
    return render_template('mobile_app.html', 
                           content=TAMIL_CONTENT if current_lang == 'ta' else {},
                           lang=current_lang)

@app.route('/reach_us/', defaults={'lang': 'en'})
@app.route('/<lang>/reach_us/', methods=['GET'])
def reach_us(lang):
    """Renders the Reach Us page with language support."""
    current_lang = lang if lang in ['en', 'ta'] else 'en'
    return render_template('reach_us.html',
                           content=TAMIL_CONTENT if current_lang == 'ta' else {},
                           lang=current_lang)

# --- Server Run Block ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)