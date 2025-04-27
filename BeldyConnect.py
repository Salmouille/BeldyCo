import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import base64
import io
import datetime

# ======================================
# 🧠 PREDICTION MODEL FUNCTIONS
# ======================================

def initialize_model():
    """Initialize or load the prediction model"""
    try:
        st.session_state.model = joblib.load('basket_predictor.joblib')
    except:
        create_sample_model()

def create_sample_model():
    """Create sample training data and train model"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Diet Type': np.random.choice(['Balanced', 'Vegetarian', 'Vegan', 'Keto'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Weekly Fats (g)': np.random.normal(70, 15, n_samples).clip(40, 100),
        'Weekly Carbs (g)': np.random.normal(1500, 300, n_samples).clip(800, 2200),
        'Weekly Proteins (g)': np.random.normal(450, 100, n_samples).clip(200, 700),
        'Weekly Fiber (g)': np.random.normal(150, 30, n_samples).clip(80, 220),
        'Item_Count': np.random.poisson(5, n_samples).clip(3, 8),
        'Has_Protein': np.random.binomial(1, 0.7, n_samples),
        'Budget': np.random.uniform(150, 350, n_samples),
        'Price': np.zeros(n_samples)
    }
    
    # Calculate prices
    for i in range(n_samples):
        base_price = 0
        if data['Diet Type'][i] == 'Keto':
            base_price += np.random.normal(200, 30)
        elif data['Diet Type'][i] == 'Vegan':
            base_price += np.random.normal(180, 25)
        else:
            base_price += np.random.normal(160, 20)
            
        base_price += data['Weekly Proteins (g)'][i] * 0.1
        base_price += data['Weekly Fats (g)'][i] * 0.05
        base_price -= data['Weekly Carbs (g)'][i] * 0.02
        base_price += data['Item_Count'][i] * 10
        
        if data['Has_Protein'][i]:
            base_price += np.random.normal(30, 5)
            
        base_price += data['Budget'][i] * 0.2
        data['Price'][i] = base_price
    
    df = pd.DataFrame(data)
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(), ['Diet Type']),
        ('num', 'passthrough', ['Weekly Fats (g)', 'Weekly Carbs (g)', 
                               'Weekly Proteins (g)', 'Weekly Fiber (g)',
                               'Item_Count', 'Has_Protein', 'Budget'])
    ])
    
    # Create and train model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    st.session_state.model = model
    joblib.dump(model, 'basket_predictor.joblib')

def predict_basket_price(diet_type, nutrition_data, selected_items, budget):
    """Predict basket price using model or fallback"""
    input_data = pd.DataFrame([{
        'Diet Type': diet_type,
        'Weekly Fats (g)': nutrition_data['fats'],
        'Weekly Carbs (g)': nutrition_data['carbs'],
        'Weekly Proteins (g)': nutrition_data['proteins'],
        'Weekly Fiber (g)': nutrition_data['fiber'],
        'Item_Count': len(selected_items),
        'Has_Protein': int(any('Chicken' in item or 'Eggs' in item or 'Milk' in item for item in selected_items)),
        'Budget': budget
    }])

    try:
        if st.session_state.model:
            predicted_price = st.session_state.model.predict(input_data)[0]
        else:
            raise ValueError("Model not loaded")
    except:
        # Fallback calculation
        base_price = sum(st.session_state.all_items.get(item, 0) for item in selected_items)
        multiplier = 1.0
        if nutrition_data['proteins'] > 150: multiplier += 0.1
        if nutrition_data['fats'] > 80: multiplier += 0.05
        if nutrition_data['carbs'] < 1000: multiplier += 0.05
        predicted_price = base_price * multiplier

    min_price = sum(st.session_state.all_items.get(item, 0) for item in selected_items) * 0.8
    max_price = sum(st.session_state.all_items.get(item, 0) for item in selected_items) * 1.5
    return max(min(predicted_price, max_price), min_price)

def log_feedback(rating, comments):
    """Log feedback to a CSV file"""
    feedback_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "username": st.session_state.username,
        "rating": rating,
        "comments": comments,
        "order_details": st.session_state.selected_basket
    }
    
    try:
        # Try to append to existing feedback file
        pd.DataFrame([feedback_data]).to_csv("feedback.csv", mode="a", header=False, index=False)
    except:
        # Create new file if doesn't exist
        pd.DataFrame([feedback_data]).to_csv("feedback.csv", index=False)

# ======================================
# 🎨 CUSTOM THEME & INITIALIZATION
# ======================================

def setup_app():
    # Custom CSS injection
    st.markdown("""
    <style>
        :root {
            --primary: #2e8b57;
            --secondary: #3a5a40;
            --accent: #588157;
            --light: #f8f9fa;
            --dark: #212529;
        }
        
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        .stApp {
            background-color: #f5f5f5;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--primary);
        }
        
        .stButton>button {
            background-color: var(--primary);
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: var(--secondary);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .basket-card {
            padding: 1.5rem; 
            border-radius: 12px; 
            background: white; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            height: 100%;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .basket-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        
        .price-prediction {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            border-left: 4px solid var(--accent);
        }
        
        .basket-image {
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 1rem;
            transition: transform 0.3s ease;
        }
        
        .basket-card:hover .basket-image {
            transform: scale(1.03);
        }
        
        .option-card {
            padding: 2rem;
            border-radius: 12px;
            background: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            height: 100%;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 1.5rem;
        }
        
        .option-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        
        .item-list {
            max-height: 300px;
            overflow-y: auto;
            margin-top: 1rem;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        .delivery-option-card {
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            background: white;
            border-left: 4px solid var(--accent);
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .delivery-option-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .timeline-item {
            padding: 0.8rem 1.2rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            display: flex;
            align-items: center;
            background: #f0f0f0;
        }
        
        .timeline-item-current {
            background: var(--primary);
            color: white;
            animation: pulse 2s infinite;
            border-left: 4px solid var(--accent);
        }
        
        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }
        
        .stTabs [role="tab"] {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .stTabs [role="tab"][aria-selected="true"] {
            background: var(--primary);
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize all session state variables with default values
    if 'initialized' not in st.session_state:
        st.session_state.update({
            'initialized': True,
            'logged_in': False,
            'username': None,
            'show_signup': False,
            'show_login': False,
            'show_feedback': False,
            'order_confirmed': False,
            'selected_basket': None,
            'order_address': "",
            'custom_basket': [],
            'show_custom_basket': False,
            'show_existing_baskets': False,
            'delivery_details': None,
            'chat_messages': [],
            'show_contact_driver': False,
            'users': {
                "student1": {
                    "password": "studentpass",
                    "first_name": "Salma",
                    "last_name": "Sabri",
                    "phone": "0628355884"
                }
            },
            'all_items': {
                'Vegetables (1kg)': 25,
                'Fruits (1kg)': 30,
                'Cheese (250g)': 25,
                'Yogurt': 10,
                'Milk (1L)': 15,
                'Chicken (1kg)': 50,
                'Eggs (dozen)': 20,
                'Bread (loaf)': 8,
                'Rice (1kg)': 20,
                'Energy Balls': 15,
                'Protein Bars': 12,
                'Dried Fruits': 18,
                'Granola Bars': 10,
                'Nuts (200g)': 25,
                'Dark Chocolate': 20
            },
            'diet_type': 'Balanced',
            'weekly_proteins': 120,
            'weekly_carbs': 300,
            'weekly_fats': 70,
            'weekly_fiber': 25,
            'budget_slider': 200,
            'model': None,
            'predicted_price': 0,
            'last_calculation_hash': None,
            'current_basket_hash': None
        })
        
        initialize_model()

# ======================================
# 🖼️ HEADER COMPONENT
# ======================================

def img_to_base64(img):
    """Convert image to base64 for HTML display"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def header_section():
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: var(--primary);'>Beldy Connect</h1>
        <p>Smart Grocery Platform for Students</p>
    </div>
    """, unsafe_allow_html=True)

    possible_paths = [
        os.path.join("beldyimages", "G.jpg"),
        os.path.join("beldyimages", "G.png"),
        os.path.join(os.getcwd(), "beldyimages", "G.jpg"),
        os.path.join(os.getcwd(), "beldyimages", "G.png"),
        "G.jpg",
        "G.png"
    ]

    img_found = None
    for path in possible_paths:
        if os.path.exists(path):
            img_found = path
            break

    if img_found:
        try:
            with open(img_found, "rb") as f:
                img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode()
                
            st.markdown(f"""
            <style>
                .dashboard-banner {{
                    width: 100%;
                    max-height: 250px;
                    object-fit: cover;
                    border-radius: 12px;
                    margin: 1rem 0;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }}
            </style>
            <img class='dashboard-banner' 
                 src='data:image/jpeg;base64,{img_b64}'
                 alt='Dashboard Banner'>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    else:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 2rem; border-radius: 8px; text-align: center;'>
            <p>Placeholder for dashboard banner</p>
        </div>
        """, unsafe_allow_html=True)

# ======================================
# 👤 AUTHENTICATION PAGES
# ======================================

def signup_page():
    with st.container():
        st.markdown("""
        <div class='card'>
            <h2 style='color: var(--primary)'>📝 Create Account</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("signup_form"):
            cols = st.columns(2)
            with cols[0]:
                first_name = st.text_input("First Name", key="signup_first_name")
            with cols[1]:
                last_name = st.text_input("Last Name", key="signup_last_name")
            
            username = st.text_input("Username", key="signup_username")
            phone = st.text_input("Phone Number", key="signup_phone")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
            
            if st.form_submit_button("Sign Up"):
                if password != confirm_password:
                    st.error("Passwords don't match!")
                elif username in st.session_state.users:
                    st.error("Username already exists!")
                else:
                    st.session_state.users[username] = {
                        "password": password,
                        "first_name": first_name,
                        "last_name": last_name,
                        "phone": phone
                    }
                    st.success("Account created! Please login.")
                    st.session_state.show_login = True
                    st.rerun()
        
        if st.button("← Back to Login"):
            st.session_state.show_signup = False
            st.rerun()

def login_page():
    with st.container():
        st.markdown("""
        <div class='card'>
            <h2 style='color: var(--primary)'>🔑 Login</h2>
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            user = st.session_state.users.get(username)
            if user and user['password'] == password:
                st.session_state.update({
                    'logged_in': True,
                    'username': username,
                    'user_info': user,
                    'show_custom_basket': False,
                    'show_existing_baskets': False,
                    'selected_basket': None
                })
                st.rerun()
            else:
                st.error("Invalid credentials")
        
        st.markdown("Don't have an account? **Sign up below**")
        if st.button("Go to Sign Up"):
            st.session_state.show_signup = True
            st.rerun()

# ======================================
# 🧺 BASKET SELECTION PAGES
# ======================================

def show_basket_options():
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>Welcome <span style="color: var(--primary);">{st.session_state.user_info['first_name']}</span>!</h1>
        <p>How would you like to create your basket?</p>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(2)
    with cols[0]:
        if st.button("🛒 Customize Your Basket", key="custom_basket_btn", use_container_width=True):
            st.session_state.update({
                'show_custom_basket': True,
                'show_existing_baskets': False,
                'selected_basket': None,
                'custom_basket': []
            })
            st.rerun()
        
        st.markdown("""
        <p style="text-align: center; margin-top: 1rem;">
            Select individual items to create your perfect basket
        </p>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        if st.button("🧺 Choose Existing Basket", key="existing_basket_btn", use_container_width=True):
            st.session_state.update({
                'show_existing_baskets': True,
                'show_custom_basket': False,
                'selected_basket': None
            })
            st.rerun()
        
        st.markdown("""
        <p style="text-align: center; margin-top: 1rem;">
            Select from our pre-designed baskets
        </p>
        """, unsafe_allow_html=True)

def show_custom_basket():
    st.markdown("""
    <div class='card'>
        <h2 style="color: var(--primary); border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem;">
            🛒 Customize Your Basket
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Nutrition Profile
    with st.expander("Nutrition Profile", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            st.session_state.diet_type = st.selectbox(
                "Diet Type", 
                ["Balanced", "Vegetarian", "Vegan", "Keto"],
                index=["Balanced", "Vegetarian", "Vegan", "Keto"].index(st.session_state.diet_type)
            )
            st.session_state.weekly_proteins = st.number_input(
                "Weekly Proteins (g)", 
                min_value=0, 
                value=st.session_state.weekly_proteins
            )
            
        with cols[1]:
            st.session_state.weekly_carbs = st.number_input(
                "Weekly Carbs (g)", 
                min_value=0, 
                value=st.session_state.weekly_carbs
            )
            st.session_state.weekly_fats = st.number_input(
                "Weekly Fats (g)", 
                min_value=0, 
                value=st.session_state.weekly_fats
            )
        
        st.session_state.weekly_fiber = st.number_input(
            "Weekly Fiber (g)", 
            min_value=0, 
            value=st.session_state.weekly_fiber
        )
    
    # Budget Slider
    with st.expander("Budget", expanded=True):
        st.session_state.budget_slider = st.slider(
            "Weekly Budget (MAD)",
            min_value=150,
            max_value=350,
            value=st.session_state.budget_slider
        )
    
    # Item Selection
    with st.expander("Select Items", expanded=True):
        st.markdown("### Available Items")
        
        new_custom_basket = []
        for item, price in st.session_state.all_items.items():
            if st.checkbox(f"{item} - {price} MAD", key=f"item_{item}"):
                new_custom_basket.append(item)
        
        st.session_state.custom_basket = new_custom_basket
        
        if st.session_state.custom_basket:
            st.markdown("### Your Selected Items")
            for item in st.session_state.custom_basket:
                st.write(f"- {item} ({st.session_state.all_items[item]} MAD)")
        else:
            st.info("No items selected yet")
    
    # Price Prediction
    if st.button("🔮 Predict Basket Price"):
        if not st.session_state.custom_basket:
            st.error("Please select at least one item!")
        else:
            nutrition_data = {
                'proteins': st.session_state.weekly_proteins,
                'carbs': st.session_state.weekly_carbs,
                'fats': st.session_state.weekly_fats,
                'fiber': st.session_state.weekly_fiber
            }
            
            predicted_price = predict_basket_price(
                st.session_state.diet_type,
                nutrition_data,
                st.session_state.custom_basket,
                st.session_state.budget_slider
            )
            
            st.session_state.predicted_price = predicted_price
            st.session_state.selected_basket = {
                "name": "Custom Basket",
                "items": st.session_state.custom_basket,
                "price": predicted_price
            }
            st.rerun()
    
    if st.button("← Back to Options"):
        st.session_state.show_custom_basket = False
        st.rerun()

def show_existing_baskets():
    st.markdown("""
    <div class='card'>
        <h2 style="color: var(--primary); border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem;">
            🧺 Choose Existing Basket
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    BASKETS = {
        "The Chef's Basket": {
            "description": "For daily cooking",
            "items": ["Vegetables (1kg)", "Fruits (1kg)", "Cheese (250g)", "Yogurt", "Milk (1L)", "Chicken (1kg)", "Eggs (dozen)", "Bread (loaf)", "Rice (1kg)"],
            "icon": "👨‍🍳"
        },
        "Snacker's Basket": {
            "description": "Healthy easy-to-eat snacks",
            "items": ["Energy Balls", "Protein Bars", "Dried Fruits", "Granola Bars", "Nuts (200g)", "Dark Chocolate"],
            "icon": "🍿"
        },
        "Balanced Basket": {
            "description": "Mix of cooking + snacks",
            "items": ["Fruits (1kg)", "Vegetables (1kg)", "Eggs (dozen)", "Yogurt", "Granola Bars", "Rice (1kg)"],
            "icon": "⚖️"
        }
    }
    
    cols = st.columns(3)
    for i, (basket_name, basket_data) in enumerate(BASKETS.items()):
        with cols[i % 3]:
            # Calculate total price for this basket
            total_price = sum(st.session_state.all_items.get(item, 0) for item in basket_data['items'])
            
            st.markdown(
                f"""
                <div class='basket-card'>
                    <div style='font-size: 2rem;'>{basket_data['icon']}</div>
                    <h3>{basket_name}</h3>
                    <p><small>{basket_data['description']}</small></p>
                    <div style='margin-top: 0.5rem; font-weight: bold; color: var(--primary);'>
                        {total_price} MAD
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if st.button(f"Select {basket_name}", key=f"select_{i}"):
                st.session_state.selected_basket = {
                    "name": basket_name,
                    "items": basket_data['items'],
                    "price": total_price
                }
                
                # Display basket details in a nice card
                st.markdown(f"""
                <div class='card' style='margin-top: 1rem; background-color: #f8f9fa;'>
                    <h3 style='color: var(--primary);'>Selected: {basket_name}</h3>
                    <div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>
                        <strong>Total Price:</strong> {total_price} MAD
                    </div>
                    <div style='margin-bottom: 0.5rem;'>
                        <strong>Includes:</strong>
                        <ul style='margin-top: 0.5rem;'>
                            {"".join([f"<li>{item} ({st.session_state.all_items.get(item, 0)} MAD)</li>" for item in basket_data['items']])}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.success(f"{basket_name} selected! Proceed to checkout below.")
    
    if st.button("← Back to Options"):
        st.session_state.show_existing_baskets = False
        st.rerun()
    

# ======================================
# 🚚 DELIVERY SYSTEM
# ======================================

def show_delivery_choice():
    st.markdown("""
    <div class='card'>
        <h2 style="color: var(--primary);">🚚 Delivery Options</h2>
    </div>
    """, unsafe_allow_html=True)
    
    delivery_method = st.radio(
        "Choose your delivery method:",
        ["📦 Traditional Delivery", "🚲 BikeSync Pickup"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if delivery_method == "📦 Traditional Delivery":
        show_traditional_delivery()
    else:
        show_bikesync_delivery()

def show_traditional_delivery():
    st.markdown("""
    <div class='card'>
        <h2 style="color: var(--primary); border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem;">
            📦 Traditional Delivery
        </h2>
        <p style="margin-top: -0.5rem; color: var(--secondary);">
            Convenient delivery to campus locations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📍 Address & Timing", "🚚 Track Delivery", "🆘 Support"])
    
    with tab1:
        address_details = show_address_verification()
        time_details = show_delivery_scheduling()
        package_details = show_package_options()
        
        additional_fees = 20 if time_details["is_express"] else 0
        additional_fees += 5 if package_details["chill_bag"] else 0
        
        st.markdown("---")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Delivery Summary**")
            st.write(f"📍 {address_details['location']}")
            st.write(f"⏰ {time_details['time_window']}")
            st.write(f"📅 {time_details['date']}")
        with cols[1]:
            st.markdown("**Additional Options**")
            st.write("🚀 Express" if time_details["is_express"] else "🐢 Standard")
            st.write("❄️ Chill bag" if package_details["chill_bag"] else "🌡️ Room temp")
            st.write(f"💵 Additional fees: {additional_fees} MAD")
        
        if st.button("✅ Confirm Traditional Delivery", type="primary"):
            st.session_state.delivery_details = {
                **address_details,
                **time_details,
                **package_details,
                "method": "Traditional",
                "fees": additional_fees
            }
            st.balloons()
            st.success("Delivery scheduled successfully!")

            if st.session_state.selected_basket:
                delivery_fees = st.session_state.delivery_details.get("fees", 0)
                total = st.session_state.selected_basket["price"] + delivery_fees
                st.info(f"🧾 **Your total: {total} MAD** (Basket: {st.session_state.selected_basket['price']} + Delivery: {delivery_fees})")

            
    
    with tab2:
        show_delivery_tracker()
    
    with tab3:
        show_support_options()

def show_bikesync_delivery():
    st.markdown("""
    <div class='card'>
        <h2 style="color: var(--primary); border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem;">
            🚲 BikeSync Pickup
        </h2>
        <p style="margin-top: -0.5rem; color: var(--secondary);">
            Eco-friendly campus bike pickup
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #e8f5e9; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <h4 style="color: var(--primary); margin-top: 0;">How BikeSync Works:</h4>
        <ol>
            <li>Click the link below to access BikeSync</li>
            <li>Book your bike pickup time</li>
            <li>Use campus e-bike </li>
            <li>Return bike when done</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Replace the pickup time selection with a link to BikeSync
    st.markdown("""
    <div style="text-align: center; margin: 1.5rem 0;">
        <a href="https://bikesyncifrane.vercel.app/" target="_blank" style="text-decoration: none;">
            <button style="background-color: var(--primary); color: white; border: none; 
                        padding: 0.75rem 1.5rem; border-radius: 8px; font-size: 1rem;
                        cursor: pointer; transition: all 0.3s ease;">
                🚴 Book BikeSync Pickup
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("✅ Confirm BikeSync Pickup", type="primary"):
        st.session_state.delivery_details = {
            "method": "BikeSync",
            "pickup_time": "Booked via BikeSync website",
            "location": "Market Hub",
            "fees": 20
        }
        st.balloons()
        st.success("BikeSync pickup confirmed! Please book your time slot on the BikeSync website.")
        if st.session_state.selected_basket:
            delivery_fees = st.session_state.delivery_details.get("fees", 0)
            total = st.session_state.selected_basket["price"] + delivery_fees
            st.info(f"🧾 **Your total: {total} MAD** (Basket: {st.session_state.selected_basket['price']} + Delivery: {delivery_fees})")

def show_address_verification():
    st.markdown("### 📍 Delivery Address")
    
    campus_buildings = {
        "Main Gate": {"description": "Primary entrance (Security booth)"},

        "Cafeteria": {"description": "Central food court"},
        "Sports Complex": {"description": "Near gym entrance"}
    }
    
    selected_location = st.selectbox(
        "Select campus delivery point:",
        options=list(campus_buildings.keys()),
        index=0,
        help="Choose where you want your groceries delivered"
    )
    
    special_notes = st.text_area(
        "Special delivery instructions (optional):",
        placeholder="e.g., 'Leave with security guard', 'Call upon arrival'",
        height=100
    )
    
    return {
        "location": selected_location,
        "notes": special_notes,
        "details": campus_buildings[selected_location]["description"]
    }

def show_delivery_scheduling():
    st.markdown("### ⏰ Delivery Time")
    
    now = datetime.datetime.now()
    time_slots = []
    
    # Generate time slots for today
    for i in range(1, 5):
        delivery_time = now + datetime.timedelta(hours=i)
        if delivery_time.hour < 22:  # Don't schedule deliveries too late
            time_slots.append(delivery_time.strftime("%I:%M %p"))
    
    # If we don't have enough time slots today, add some default ones
    while len(time_slots) < 4:
        # Add default time slots in 2-hour increments starting from current time
        default_time = now + datetime.timedelta(hours=len(time_slots)+1)
        time_slots.append(default_time.strftime("%I:%M %p"))
    
    delivery_option = st.radio(
        "Choose delivery speed:",
        options=[
            f"🚀 Express ({time_slots[0]} - {time_slots[1]}) +20 MAD", 
            f"🐢 Standard ({time_slots[2]} - {time_slots[3]}) Free"
        ],
        index=1
    )

    delivery_date = st.date_input(
        "Select delivery date:",
        min_value=datetime.date.today(),
        max_value=datetime.date.today() + datetime.timedelta(days=7),
        format="YYYY-MM-DD"
    )
    
    return {
        "is_express": "Express" in delivery_option,
        "time_window": delivery_option.split("(")[1].split(")")[0],
        "date": delivery_date.strftime("%A, %B %d")
    }

def show_package_options():
    st.markdown("### 📦 Package Handling")
    
    cols = st.columns(2)
    with cols[0]:
        packaging = st.radio(
            "Packaging type:",
            options=["♻️ Eco-friendly (Cardboard)", "🛍️ Standard plastic"],
            index=0
        )
    
    with cols[1]:
        temperature = st.radio(
            "Temperature control:",
            options=["❄️ Chill bag (+5 MAD)", "🌡️ Room temperature"],
            index=1
        )
    
    signature = st.checkbox(
        "📝 Require signature confirmation",
        value=True,
        help="For security, we recommend keeping this enabled"
    )
    
    return {
        "eco_packaging": "Eco-friendly" in packaging,
        "chill_bag": "Chill bag" in temperature,
        "signature_required": signature
    }

def show_delivery_tracker():
    st.markdown("### 🚚 Delivery Status")
    
    stages = [
        {"name": "Order Confirmed", "icon": "✓", "time": "10:00 AM", "status": "complete"},
        {"name": "Packing", "icon": "📦", "time": "10:15 AM", "status": "complete"},
        {"name": "Quality Check", "icon": "🔍", "time": "10:30 AM", "status": "complete"},
        {"name": "Dispatched", "icon": "🛵", "time": "11:00 AM", "status": "current"},
        {"name": "In Transit", "icon": "🚚", "time": "11:30 AM", "status": "pending"},
        {"name": "Arriving Soon", "icon": "📲", "time": "", "status": "pending"}
    ]
    
    for stage in stages:
        is_current = stage["status"] == "current"
        bg_color = "var(--primary)" if is_current else "#f0f0f0"
        text_color = "white" if is_current else "var(--dark)"
        animation = "animation: pulse 2s infinite;" if is_current else ""
        
        st.markdown(f"""
        <div style="background: {bg_color}; color: {text_color}; 
                    padding: 0.8rem 1.2rem; border-radius: 12px; 
                    margin: 0.5rem 0; display: flex; 
                    align-items: center; {animation}
                    border-left: 4px solid {"var(--accent)" if is_current else "transparent"}">
            <div style="font-size: 1.5rem; margin-right: 1.5rem; width: 30px;">{stage['icon']}</div>
            <div style="flex-grow: 1;">
                <div style="font-weight: 600; font-size: 1.1rem;">{stage['name']}</div>
                <div>{stage['time']}</div>
            </div>
            {"" if not is_current else "<div style='font-size: 0.9rem;'>LIVE</div>"}
        </div>
        """, unsafe_allow_html=True)
    
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 2rem; margin-right: 1rem;">👨🏽‍💼</div>
                <div>
                    <strong>Driver: Youssef</strong>
                    <div>Vehicle: Scooter (Blue)</div>
                    <div>Plate: ABC-1234</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        if st.button("📞 Call Driver", use_container_width=True):
            st.session_state.show_contact_driver = True

def show_support_options():
    st.markdown("### 🆘 Need Help?")
    
    tab1, tab2, tab3 = st.tabs(["📞 Call", "💬 Chat", "📧 Email"])
    
    with tab1:
        st.info("""
        **Delivery Support Hotline**  
        +212 522-123456  
        Available 7AM-11PM
        """)
        if st.button("Call Now", key="call_support"):
            st.success("Connecting you to our support team...")
    
    with tab2:
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
            
        for msg in st.session_state.chat_messages:
            st.chat_message(msg["role"]).write(msg["content"])
        
        if prompt := st.chat_input("Type your question"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            st.rerun()
    
    with tab3:
        with st.form("email_form"):
            subject = st.selectbox("Issue type", ["Late delivery", "Missing items", "Quality issue", "Other"])
            message = st.text_area("Details", height=150)
            if st.form_submit_button("Send Email"):
                st.success("Support ticket #2023 created!")

# ======================================
# 📱 STUDENT DASHBOARD
# ======================================

def student_dashboard():
    # Prevent crash if delivery details are missing
    delivery_details = st.session_state.delivery_details
    selected_basket = st.session_state.selected_basket

    if st.session_state.order_confirmed:
        if not delivery_details:
            st.warning("Please select delivery method first")
            show_delivery_choice()
            return

        delivery_fees = delivery_details.get('fees', 0)
        delivery_method = delivery_details.get('method', 'N/A')
        delivery_address = st.session_state.order_address or "N/A"
        total_price = selected_basket['price'] + delivery_fees

        st.markdown(f"""
        <div class='card' style='background-color: #e8f5e9;'>
            <h2 style='color: var(--primary); text-align: center;'>🎉 Thank you for your order!</h2>
            <div class='card' style='margin: 1rem 0; background: white;'>
                <h3 style='color: var(--primary); border-bottom: 1px solid #eee; padding-bottom: 0.5rem;'>Order Summary</h3>
                <div style='display: flex; justify-content: space-between;'>
                    <div>
                        <h4>{selected_basket['name']}</h4>
                        <ul style='margin-top: 0.5rem;'>
                            {"".join([f"<li>{item}</li>" for item in selected_basket['items']])}
                        </ul>
                    </div>
                    <div style='text-align: right;'>
                        <p><strong>Subtotal:</strong> {selected_basket['price']} MAD</p>
                        <p><strong>Delivery Fee:</strong> {delivery_fees} MAD</p>
                        <p style='font-size: 1.2rem; font-weight: bold; margin-top: 1rem;'>Total: {total_price} MAD</p>
                    </div>
                </div>
            </div>
            <p style='text-align: center;'>Delivery Method: {delivery_method}</p>
            <p style='text-align: center;'>You will be delivered soon to:</p>
            <p style='text-align: center; font-weight: bold;'>{delivery_address}</p>
        </div>
        """, unsafe_allow_html=True)

        # Delivery confirmation button
        if not st.session_state.get("delivery_completed"):
            if st.button("✅ I received my basket"):
                st.session_state.delivery_completed = True
                st.rerun()

        # Feedback form after delivery
        if st.session_state.get("delivery_completed"):
            st.markdown("### 💬 We'd love your feedback!")
            with st.form("post_delivery_feedback_form"):
                rating = st.slider("Rate your experience (1 = Bad, 5 = Excellent)", 1, 5, 5)
                comments = st.text_area("Any suggestions or issues you'd like to share?")
                if st.form_submit_button("Submit Feedback"):
                    log_feedback(rating, comments)
                    st.success("Thanks for helping us improve!")

        show_delivery_choice()

        if st.button("Start New Order"):
            st.session_state.update({
                'order_confirmed': False,
                'selected_basket': None,
                'show_custom_basket': False,
                'show_existing_baskets': False,
                'delivery_details': None,
                'delivery_completed': False
            })
            st.rerun()

        return

    # Not yet ordered: show basket selection flows
    if st.session_state.show_custom_basket:
        show_custom_basket()
    elif st.session_state.show_existing_baskets:
        show_existing_baskets()
    else:
        show_basket_options()

    if selected_basket:
        show_order_confirmation()

def show_order_confirmation():
    st.markdown("---")
    st.markdown("""
    <div class='card'>
        <h2 style="color: var(--primary);">✅ Confirm Your Order</h2>
    </div>
    """, unsafe_allow_html=True)
    
    address = st.text_input("Delivery Address", key="delivery_address")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm Order", use_container_width=True):
            if address.strip() == "":
                st.error("Please enter your address!")
            elif not st.session_state.selected_basket:
                st.error("Please select a basket first!")
            else:
                st.session_state.order_confirmed = True
                st.session_state.order_address = address
                st.rerun()
    
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.show_feedback = True
            st.session_state.logged_in = False
            st.rerun()
    
    if st.session_state.show_feedback:
        with st.form("feedback_form"):
            feedback_msg = st.text_area("If you didn't like something, tell us here!!", key="feedback_msg")
            if st.form_submit_button("Submit Feedback"):
                st.success("Thank you for your feedback!")
                st.session_state.show_feedback = False
                st.rerun()

# ======================================
# 🚀 MAIN APP FLOW
# ======================================

def main():
    setup_app()
    header_section()
    
    if not st.session_state.logged_in:
        if st.session_state.show_signup:
            signup_page()
        else:
            login_page()
    else:
        student_dashboard()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: var(--dark); padding: 1rem'>
        <p>🌿 Beldy Connect - Fair Prices for Students</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
