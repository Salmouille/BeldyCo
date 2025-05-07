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

st.set_page_config(
    page_title="Beldy Connect",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASKETS = {
    "The Chef's Basket": {
        "items": ["Vegetables (1kg)", "Fruits (1kg)", "Cheese (250g)", "Yogurt", "Milk (1L)", "Chicken (1kg)", "Eggs (dozen)", "Bread (loaf)", "Rice (1kg)"],
        "image": "chefs_basket.jpg"
    },
    "Snacker's Basket": {
        "items": ["Energy Balls", "Protein Bars", "Dried Fruits", "Granola Bars", "Nuts (200g)", "Dark Chocolate"],
        "image": "snacker.jpg"
    },
    "Balanced Basket": {
        "items": ["Fruits (1kg)", "Vegetables (1kg)", "Eggs (dozen)", "Yogurt", "Granola Bars", "Rice (1kg)"],
        "image": "balanced.jpg"
    },
    "Breakfast Basket": {
        "items": ["Homemade Bread", "Homemade Jam", "Fresh Butter", "Local Honey", "Farm Eggs (dozen)", "Fresh Cheese","Homemade Hricha", "Local Tea Herbs", "Homemade Ground Coffee", "Fresh Cow/Goat Milk" ],
        "image": "breakfast.jpg",
        "fixed_price": 100  
    }
}


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
    # Initialize all session state variables with default values
    if 'initialized' not in st.session_state:
        st.session_state.update({
            'initialized': True,
            'logged_in': False,
            'username': None,
            'show_signup': False,
            'show_login': True,
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
            'delivery_completed': False,
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
                'Dark Chocolate': 20,
                'Homemade Bread': 25,
                'Homemade Jam': 20,
                'Fresh Butter': 15,
                'Local Honey': 30,
                'Farm Eggs (dozen)': 25,
                'Fresh Cheese': 20,
                'Homemade Hricha': 15,  
                'Local Tea Herbs': 10,  
                'Homemade Ground Coffee': 25, 
                'Fresh Cow/Goat Milk': 20  

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
    st.title("Beldy Connect")
    st.write("Smart Grocery Platform for Students")

    possible_paths = [
        os.path.join("imgs", "logo.jpg"),
        os.path.join("imgs", "logo.png"),
        os.path.join(os.getcwd(), "imgs", "logo.jpg"),
        os.path.join(os.getcwd(), "imgs", "logo.png"),
        "logo.jpg",
        "logo.png"
    ]

    img_found = None
    for path in possible_paths:
        if os.path.exists(path):
            img_found = path
            break

    if img_found:
        try:
            image = Image.open(img_found)
            st.image(image, caption='', use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    else:
        st.write("Welcome to Beldy Connect - Fresh groceries delivered to your campus")

# ======================================
# 👤 AUTHENTICATION PAGES
# ======================================

def signup_page():
    st.subheader("📝 Create Account")
        
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
        
        submitted = st.form_submit_button("Sign Up")
        if submitted:
            if not first_name or not last_name or not username or not phone or not password:
                st.error("Please fill out all fields")
            elif password != confirm_password:
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
                st.session_state.show_signup = False
                st.rerun()
    
    if st.button("← Back to Login"):
        st.session_state.show_signup = False
        st.session_state.show_login = True
        st.rerun()

def login_page():
    st.subheader("🔑 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        submitted = st.form_submit_button("Login")
        if submitted:
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
                st.success(f"Welcome back, {user['first_name']}!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    st.write("Don't have an account? **Sign up below**")
    if st.button("Go to Sign Up"):
        st.session_state.show_signup = True
        st.session_state.show_login = False
        st.rerun()

# ======================================
# 🧺 BASKET SELECTION PAGES
# ======================================

def show_basket_options():
    st.write(f"Welcome {st.session_state.user_info['first_name']}!")
    st.write("How would you like to create your basket?")
    
    cols = st.columns(2)
    with cols[0]:
        st.write("🛒")
        st.subheader("Customize Your Basket")
        st.write("Select individual items to create your perfect basket")
        if st.button("Create Custom Basket", key="custom_basket_btn"):
            st.session_state.update({
                'show_custom_basket': True,
                'show_existing_baskets': False,
                'selected_basket': None,
                'custom_basket': []
            })
            st.rerun()
    
    with cols[1]:
        st.write("🧺")
        st.subheader("Choose Existing Basket")
        st.write("Select from our pre-designed baskets")
        if st.button("Browse Baskets", key="existing_basket_btn"):
            st.session_state.update({
                'show_existing_baskets': True,
                'show_custom_basket': False,
                'selected_basket': None
            })
            st.rerun()

def show_custom_basket():
    st.subheader("🛒 Customize Your Basket")
    
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
        st.write("### Available Items")
        
        item_cols = st.columns(3)
        new_custom_basket = []
        
        for i, (item, price) in enumerate(st.session_state.all_items.items()):
            with item_cols[i % 3]:
                if st.checkbox(f"{item} - {price} MAD", key=f"item_{item}"):
                    new_custom_basket.append(item)
        
        st.session_state.custom_basket = new_custom_basket
        
        if st.session_state.custom_basket:
            st.write("### Your Selected Items")
            total = 0
            for item in st.session_state.custom_basket:
                price = st.session_state.all_items[item]
                total += price
                st.write(f"- {item} ({price} MAD)")
            
            st.info(f"Estimated base price: {total} MAD")
        else:
            st.info("No items selected yet")
    
    # Price Prediction
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    with col2:
        if st.button("← Back to Options"):
            st.session_state.show_custom_basket = False
            st.rerun()

def show_existing_baskets():
    st.subheader("🧺 Choose Existing Basket")
    st.write("Select one of our carefully curated baskets")

    cols = st.columns(4)  # Changed to 4 columns to accommodate new basket
    for i, (name, data) in enumerate(BASKETS.items()):
        with cols[i % 4]:  # Changed to 4 columns
            # Set default image path
            basket_img = os.path.join("imgs", "default_basket.jpg")
            
            if name == "The Chef's Basket":
                basket_img = os.path.join("imgs", "chef.jpg")
            elif name == "Snacker's Basket":
                basket_img = os.path.join("imgs", "snack.jpg")
            elif name == "Balanced Basket":
                basket_img = os.path.join("imgs", "snacks.jpg")
            elif name == "Breakfast Basket":
                basket_img = os.path.join("imgs", "br.jpg")

            # Try to load the image with error handling
            try:
                st.image(basket_img, caption=name, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading basket image: {str(e)}")
                st.image(os.path.join("imgs", "default_basket.jpg"), 
                         caption="Basket Image", 
                         use_container_width=True)

            # Use fixed price if specified, otherwise calculate
            price = data.get('fixed_price', sum(st.session_state.all_items.get(item, 0) for item in data['items']))
            
            st.subheader(name)
            st.write(f"**{price} MAD**")
            
            # Display all items without "more items" message
            for item in data['items']:
                st.write(f"- {item}")

            if st.button(f"Select {name}", key=f"btn_{i}"):
                st.session_state.selected_basket = {
                    "name": name,
                    "items": data['items'],
                    "price": price,
                    "image": basket_img
                }
                st.success(f"{name} selected!")
                st.rerun()

    if st.button("← Back"):
        st.session_state.show_existing_baskets = False
        st.rerun()
# ======================================
# 🚚 DELIVERY SYSTEM
# ======================================

def show_delivery_choice():
    if not st.session_state.selected_basket:
        st.error("Please select a basket first!")
        return
        
    st.subheader("🚚 Choose Delivery Method")
    
    delivery_method = st.radio(
        "Choose your delivery method:",
        ["📦 Traditional Delivery", "🚲 BikeSync Pickup"],
        horizontal=True
    )
    
    if delivery_method == "📦 Traditional Delivery":
        show_traditional_delivery()
    else:
        show_bikesync_delivery()

def show_order_summary():
    basket = st.session_state.selected_basket
    delivery_fees = st.session_state.delivery_details.get('fees', 0) if st.session_state.get('delivery_details') else 0
    
    st.subheader("Order Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{basket['name']}**")
        for item in basket['items']:  # Removed the slicing to show all items
            st.write(f"- {item}")
    
    with col2:
        st.write(f"**Subtotal:** {basket['price']} MAD")
        if st.session_state.get('delivery_details'):
            st.write(f"**Delivery Fee:** {delivery_fees} MAD")
        else:
            st.write("Delivery fee will be calculated")
        st.write(f"**Total:** {basket['price'] + delivery_fees} MAD")

def show_order_confirmation():
    st.subheader("✅ Confirm Your Order")
    
    address = st.text_input("Delivery Address", key="delivery_address")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Place Order", type="primary"):
            if not address.strip():
                st.error("Please enter your delivery address!")
            elif not st.session_state.get('delivery_details'):
                st.error("Please select a delivery method!")
            else:
                st.session_state.order_confirmed = True
                st.session_state.order_address = address
                st.balloons()
                st.success("Order placed successfully!")
                st.rerun()
    
    with col2:
        if st.button("← Back"):
            st.session_state.update({
                'delivery_details': None
            })
            st.rerun()

def show_traditional_delivery():
    st.subheader("📦 Traditional Delivery")
    st.write("Convenient delivery to campus locations")
    
    tab1, tab2, tab3 = st.tabs(["📍 Address & Timing", "🚚 Track Delivery", "🆘 Support"])
    
    with tab1:
        address_details = show_address_verification()
        time_details = show_delivery_scheduling()
        package_details = show_package_options()
        
        additional_fees = 20 if time_details["is_express"] else 0
        additional_fees += 5 if package_details["chill_bag"] else 0
        
        st.write("---")
        cols = st.columns(2)
        with cols[0]:
            st.write("**Delivery Summary**")
            st.write(f"📍 {address_details['location']}")
            st.write(f"⏰ {time_details['time_window']}")
            st.write(f"📅 {time_details['date']}")
        with cols[1]:
            st.write("**Additional Options**")
            st.write("🚀 Express" if time_details["is_express"] else "🐢 Standard")
            st.write("❄️ Chill bag" if package_details["chill_bag"] else "🌡️ Room temp")
            st.write(f"💵 Additional fees: {additional_fees} MAD")
        
        if st.button("✅ Confirm Delivery Options", type="primary"):
            st.session_state.delivery_details = {
                **address_details,
                **time_details,
                **package_details,
                "method": "Traditional",
                "fees": additional_fees
            }
            st.success("Delivery options selected!")
            st.rerun()
    
    with tab2:
        show_delivery_tracker()
    
    with tab3:
        show_support_options()

def show_bikesync_delivery():
    st.subheader("🚲 BikeSync Pickup")
    st.write("Eco-friendly campus bike pickup")

    st.info("""
    **How BikeSync Works:**
    1. Click the link below to access BikeSync
    2. Book your bike pickup time
    3. Use campus e-bike
    4. Return bike when done
    
    💚 Save 10% on your next order when you use BikeSync!
    """)
    
    st.link_button("🚴 Book BikeSync Pickup", "https://bikesyncifrane.vercel.app/")
    
    pickup_location = st.selectbox(
        "Select pickup location:",
        ["Market Hub", "Campus Center", "Administrative Building"],
        index=0
    )
    
    if st.button("✅ Confirm BikeSync Pickup", type="primary"):
        st.session_state.delivery_details = {
            "method": "BikeSync",
            "pickup_time": "Booked via BikeSync website",
            "location": pickup_location,
            "fees": 10  # Lower fee for eco-friendly option
        }
        st.success("BikeSync pickup confirmed! Please book your time slot on the BikeSync website.")
        st.rerun()

def show_address_verification():
    st.write("### 📍 Delivery Address")
    
    campus_buildings = {
        "Main Gate": {"description": "Primary entrance (Security booth)"},
        "Cafeteria": {"description": "Central food court"},
        "Sports Complex": {"description": "Near gym entrance"},
        "Library": {"description": "Front desk"},
        "Student Center": {"description": "Information desk"}
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
    st.write("### ⏰ Delivery Time")
    
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
        value=datetime.date.today(),
        format="YYYY-MM-DD"
    )
    
    return {
        "is_express": "Express" in delivery_option,
        "time_window": delivery_option.split("(")[1].split(")")[0],
        "date": delivery_date.strftime("%A, %B %d")
    }

def show_package_options():
    st.write("### 📦 Package Handling")
    
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
    st.write("### 🚚 Delivery Status")
    
    if not st.session_state.order_confirmed:
        st.info("Your delivery status will appear here after you place an order.")
        return
    
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
        is_complete = stage["status"] == "complete"
        
        if is_complete:
            st.success(f"{stage['icon']} {stage['name']} - {stage['time']}")
        elif is_current:
            st.info(f"{stage['icon']} {stage['name']} - {stage['time']} (LIVE)")
        else:
            st.write(f"{stage['icon']} {stage['name']} - {stage['time']}")
    
    cols = st.columns([3, 1])
    with cols[0]:
        st.write("**Driver: Youssef**")
        st.write("Vehicle: Scooter (Blue)")
        st.write("Plate: ABC-1234")
    
    with cols[1]:
        if st.button("📞 Call Driver"):
            st.session_state.show_contact_driver = True
            st.info("Connecting to driver...")

def show_support_options():
    st.write("### 🆘 Need Help?")
    
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
        
        # If user hasn't sent any messages, show a sample conversation
        if len(st.session_state.chat_messages) == 0:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "👋 Hello! How can I help you with your order today?"}
            ]
            st.rerun()
        
        if prompt := st.chat_input("Type your question"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Simulate assistant response
            if "delivery" in prompt.lower():
                response = "Your order is currently being prepared! You'll receive a notification when it's on the way. Let me know if you need anything else!"
            elif "cancel" in prompt.lower():
                response = "I'd be happy to help you with cancellation. Please note that orders can only be cancelled within 30 minutes of placing them. Would you like to proceed?"
            elif "time" in prompt.lower() or "when" in prompt.lower():
                response = "Your delivery is scheduled for today between 2:00 PM - 4:00 PM. Our driver will call you when they're close!"
            else:
                response = "Thanks for your message! A customer service representative will respond shortly. Is there anything specific about your order you'd like help with?"
                
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with tab3:
        with st.form("email_form"):
            subject = st.selectbox("Issue type", ["Late delivery", "Missing items", "Quality issue", "Other"])
            message = st.text_area("Details", height=150)
            if st.form_submit_button("Send Email"):
                st.success("Support ticket #2023 created! Our team will get back to you shortly.")

# ======================================
# 📱 STUDENT DASHBOARD
# ======================================

def student_dashboard():
    if st.session_state.get('order_confirmed'):
        show_confirmed_order_dashboard()
    else:
        show_order_creation_dashboard()

def show_confirmed_order_dashboard():
    """Dashboard view after order has been confirmed"""
    delivery_details = st.session_state.get('delivery_details') or {}
    selected_basket = st.session_state.get('selected_basket') or {}
    
    delivery_fees = delivery_details.get('fees', 0)
    delivery_method = delivery_details.get('method', 'N/A')
    delivery_address = st.session_state.get('order_address') or "N/A"
    total_price = selected_basket.get('price', 0) + delivery_fees

    st.success("🎉 Thank you for your order!")
    
    st.write("### Order Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{selected_basket.get('name', '')}**")
        for item in selected_basket.get('items', []):  # Removed the slicing
            st.write(f"- {item}")
    
    with col2:
        st.write(f"**Subtotal:** {selected_basket.get('price', 0)} MAD")
        st.write(f"**Delivery Fee:** {delivery_fees} MAD")
        st.write(f"**Total:** {total_price} MAD")
    
    st.write(f"**Delivery Method:** {delivery_method}")
    st.write(f"**Delivery Address:** {delivery_address}")

    # Order Tracking Section
    st.write("### 🚚 Track Your Order")
    
    # Call the delivery tracker function to show status
    show_delivery_tracker()

    # Delivery confirmation button
    if not st.session_state.get("delivery_completed"):
        if st.button("✅ I received my basket", type="primary"):
            st.session_state.delivery_completed = True
            st.balloons()
            st.success("Great! We hope you enjoy your groceries!")
            st.rerun()

    # Feedback form after delivery
    if st.session_state.get("delivery_completed"):
        st.write("### 💬 We'd love your feedback!")
        
        with st.form("post_delivery_feedback_form"):
            rating = st.slider("Rate your experience (1 = Poor, 5 = Excellent)", 1, 5, 5)
            comments = st.text_area("Any suggestions or issues you'd like to share?")
            if st.form_submit_button("Submit Feedback"):
                log_feedback(rating, comments)
                st.success("Thanks for helping us improve!")

    # Start a new order button
    if st.button("🛒 Start New Order"):
        st.session_state.update({
            'order_confirmed': False,
            'selected_basket': None,
            'show_custom_basket': False,
            'show_existing_baskets': False,
            'delivery_details': None,
            'delivery_completed': False
        })
        st.rerun()

def show_order_creation_dashboard():
    """Dashboard view for creating a new order"""
    # If basket is selected but not confirmed
    if st.session_state.selected_basket and not st.session_state.order_confirmed:
        # Show order summary
        show_order_summary()
        
        # Show delivery options if not yet selected
        if not st.session_state.get('delivery_details'):
            show_delivery_choice()
        else:
            # Show final confirmation if delivery is selected
            show_order_confirmation()
    
    # Not yet ordered: show basket selection flows
    elif st.session_state.get('show_custom_basket'):
        show_custom_basket()
    elif st.session_state.get('show_existing_baskets'):
        show_existing_baskets()
    else:
        show_basket_options()

# ======================================
# 🚀 MAIN APP FLOW
# ======================================

def main():
    setup_app()
    header_section()

    if not st.session_state.get('logged_in'):
        if st.session_state.get('show_signup'):
            signup_page()
        else:
            login_page()
    else:
        student_dashboard()

    # Sidebar only appears if user is logged in
    if st.session_state.get('logged_in'):
        with st.sidebar:
            user_info = st.session_state.get('user_info', {})
            st.write(f"👤 {user_info.get('first_name', '')} {user_info.get('last_name', '')}")
            st.write("Welcome back to Beldy Connect!")

            selected_basket = st.session_state.get('selected_basket')
            delivery_details = st.session_state.get('delivery_details') or {}

            if selected_basket:
                delivery_fees = delivery_details.get('fees', 0)
                total = selected_basket.get('price', 0) + delivery_fees
                st.write("### Order Summary")
                st.write(f"**Items:** {len(selected_basket.get('items', []))}")
                st.write(f"**Subtotal:** {selected_basket.get('price', 0)} MAD")
                st.write(f"**Delivery:** {delivery_fees} MAD")
                st.write(f"**Total:** {total} MAD")
            else:
                st.write("### Order Summary")
                st.write("No items selected yet")

            # Help and Support
            st.write("### 💬 Need Help?")
            st.write("Contact us at:")
            st.write("help@beldyconnect.com")
            st.write("+212 522-123456")

            # Logout Button
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.rerun()

            st.write("---")
            st.write("🌿 Beldy Connect - Fair Prices for Students")

if __name__ == "__main__":
    main()
