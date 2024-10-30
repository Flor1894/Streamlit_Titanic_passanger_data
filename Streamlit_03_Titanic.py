import streamlit as st 
import pandas as pd
import copy
import plotly.express as px
from titanic_ai_model import get_model


st.set_page_config(
    page_title = "Conquer Blocks - Titanic",
    page_icon = "🧊",
    layout = "wide",
    initial_sidebar_state="expanded",
)

if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
    
if 'df_fil' not in st.session_state:
    st.session_state['df_fil'] = pd.DataFrame()

def clean_data(df):
    
    survived_dict = {1: "Yes", 0: "No"}
    pclass_dict = {1: "1st", 2: "2nd", 3: "3rd"}
    embarkment_dict = {"C" : "Cherbourg", "Q": "Queenstown", "S": "Southhampton"}
    
    df.replace({"Survived": survived_dict}, inplace = True)
    df.replace({"Pclass": pclass_dict}, inplace=True)
    df.replace({"Embarked": embarkment_dict}, inplace = True)
    
    df.dropna(subset = ['Fare'], inplace = True)
    df.dropna(subset = ['Age'], inplace = True)
    df.dropna(subset = ['Embarked'], inplace = True)
    
    df['count'] = 1
    
    return df

@st.cache_data
def get_data(url_s3):
    
    df = pd.read_csv(url_s3)
    df = clean_data(df)
    
    return df

@st.cache_data
def get_values(col):
    
    return sorted(st.session_state['df'][col].unique())

def update_df():
    
    st.session_state['df_fil'] = st.session_state['df'][
        (st.session_state["df"]["Survived"].isin(st.session_state["Survived"])) &
        (st.session_state["df"]["Pclass"].isin(st.session_state["Pclass"])) &
        (st.session_state["df"]["Sex"].isin(st.session_state["Sex"])) &
        (st.session_state["df"]["Embarked"].isin(st.session_state["Embarked"])) & 
        ((st.session_state["df"]["Age"] >= st.session_state["Age"][0]) & (st.session_state["df"]["Age"] <= st.session_state["Age"][1])) &
        ((st.session_state["df"]["SibSp"] >= st.session_state["SibSp"][0]) & (st.session_state["df"]["SibSp"] <= st.session_state["SibSp"][1])) &
        ((st.session_state["df"]["Parch"] >= st.session_state["Parch"][0]) & (st.session_state["df"]["Parch"] <= st.session_state["Parch"][1])) &
        ((st.session_state["df"]["Fare"] >= st.session_state["Fare"][0]) & (st.session_state["df"]["Fare"] <= st.session_state["Fare"][1])) 
    ]

def generate_plot(var1, var2, var3, color_var, num_var, plot_type):
    if plot_type == "Bar":
    
        fig = px.bar(
                st.session_state['df_fil'],
                x = var1,
                y = num_var,
                color = color_var,
                )
                
    elif plot_type == "Pie":
    
        fig = px.pie(
                st.session_state['df_fil'],
                values = num_var,
                names = var1,
                )
                
    elif plot_type == "Scatter":
    
        fig = px.scatter(
                st.session_state['df_fil'],
                x = var1,
                y = var2,
                size = num_var,
                color=color_var,
                )
                
    elif plot_type == "Heatmap":
    
        fig = px.density_heatmap(
                st.session_state['df_fil'],
                x = var1,
                y = var2,
                z = num_var,
                text_auto=True,
                )
                
    elif plot_type == "Treemap":
    
        fig = px.treemap(
                st.session_state['df_fil'],
                path = [var1, var2, var3],
                values = num_var,
                color = color_var,
                )
                
    return fig


st.header("Titanic passenger data")

url_s3 = r"https://conquerblocks-streamlit.s3.eu-west-2.amazonaws.com/titanic-train.csv"  
st.session_state['df'] = get_data(url_s3)


def page_1():
    st.subheader("⛵ Data description")
    st.markdown(
        open(r"./html_titanic_table_descripcion.html").read(),
        unsafe_allow_html = True,
    )
    
    with st.expander("Dataframe"):
        st.write(st.session_state['df'])
        
    with st.expander("Describe"):
        st.write(st.session_state['df'].describe())
    
def page_2():
    st.subheader("📈 Data analytics")
    
    if st.session_state['df_fil'].empty:
        st.session_state['df_fil'] = copy.copy(st.session_state['df'])
    
    col_plot_1, col_plot_2 = st.columns([5,1])
    
    with col_plot_1:
        
        fig_plot =st.empty()
        
    with col_plot_2:
        
        plot_type= st.selectbox(
            "Type of plot",
            options =["Bar", "Pie", "Scatter", "Heatmap","Treemap"]
        )
        
        var1 = st.selectbox(
            "1st variable",
            options = st.session_state['df_fil'].columns,
        )
        
        num_var = st.selectbox(
            "Numeric varriable",
            options = ['count', 'Fare', 'Age', 'Sibsp', 'Parch']
        )
        
        color_var = st.selectbox(
            "Color variable", 
            options = st.session_state['df_fil'].columns,
        )
        
        var2 = st.selectbox(
            "2nd variable",
            options = st.session_state['df_fil'].columns,
        )
        
        var3 = st.selectbox(
            "3rth variable",
            options = st.session_state['df_fil'].columns,
        )
        
    with st.expander("Filters"):
        
        with st.form(key="filter_form"):
    
            col_fil_1, col_fil_2 = st.columns([1, 1])
            
            with col_fil_1:
                surv_values = get_values("Survived")
                sel_surv = st.multiselect(
                    'Survived',
                    options= surv_values,
                    help = "Survived Yes or No",
                    default = surv_values,
                    key="Survived",
                )
                
                pclass_values = get_values("Pclass")
                sel_surv = st.multiselect(
                    'Pclass',
                    options= pclass_values,
                    help = "Survived Yes or No",
                    default = pclass_values,
                    key="Pclass",
                )
                
                sex_values = get_values("Sex")
                sel_sex = st.multiselect(
                    'Sex',
                    options= sex_values,
                    help = "Female or male",
                    default = sex_values,
                    key="Sex",
                )
                
                embark_values = get_values("Embarked")
                sel_embrak = st.multiselect(
                    'Embarked',
                    options= embark_values,
                    help = "Embarked port",
                    default = embark_values,
                    key="Embarked",
                )
            
            with col_fil_2:
                
                age_values = get_values("Age")
                sel_age = st.slider(
                    "Age",
                    min_value = min(age_values),
                    max_value = max(age_values),
                    value = [min(age_values),max(age_values)],
                    key = "Age",
                )
                
                sibsp_values = get_values("SibSp")
                sel_sibsp = st.slider(
                    'SibSp',
                    min_value = min(sibsp_values),
                    max_value = max(sibsp_values),
                    value = [min(sibsp_values), max(sibsp_values)],
                    help = "n of siblings / spouses aboard the Titanic",
                    key = "SibSp",
                )
                        
                parch_values = get_values("Parch")
                sel_parch = st.slider(
                    "Parch",
                    min_value = min(parch_values),
                    max_value = max(parch_values),
                    value = [min(parch_values),max(parch_values)],
                    key = "Parch",
                )
                
                fare_value = get_values("Fare")
                sel_fare = st.slider(
                    "Fare",
                    min_value = min(fare_value),
                    max_value = max(fare_value),
                    value = [min(fare_value),max(fare_value)],
                    key = "Fare",
                )
                
                submit = st.form_submit_button("Update")
        
        if submit:
            update_df()
    
    fig = generate_plot(var1, var2, var3, color_var, num_var, plot_type)
    fig_plot.write(fig)      
            
def page_3():    
    st.subheader("🤖 Artificial Intelligence")
    
    model = get_model()
    
    with st.form("prediction form"):
        col_pred_1, col_pred_2, col_pred_3 = st.columns([1,1,1])
    
        with col_pred_1:
            class_input = st.selectbox(
                    "Class",
                    options = [1,2,3]
                    )
                    
            age_input = st.number_input(
                    "Age",
                    min_value = 0.0,
                    max_value = 100.0,
                    )
                    
            sibsp_input = st.number_input(
                    "Siblings",
                    min_value = 0,
                    max_value = 10,
                    )
                    
        with col_pred_2:
            parch_input = st.number_input(
                    "Parents/Children",
                    min_value = 0,
                    max_value = 10,
                    )
                    
            fare_input = st.number_input(
                    "Fare",
                    min_value = 0.0,
                    max_value = 1000.0,
                    )
        
        with col_pred_3:        
            sex_input = st.toggle(
                    "Sex male",
                    )
                    
            q_input = st.toggle(
                    "Queenstown embarked",
                    )
                    
            s_input = st.toggle(
                    "Southhampton embarked",
                    )

            submit_prediction = st.form_submit_button("Submit prediction")
        
        if submit_prediction:
            
            input_vector = [[
                    class_input,
                    age_input,
                    sibsp_input,
                    parch_input,
                    fare_input,
                    sex_input,
                    q_input,
                    s_input,
                    ]]
            
            y_pred = model.predict(input_vector)

            if y_pred:
                
                st.success("The passenger is likely to survive!")
                
            else:
                
                st.error("The passenger is likely to die...")


pg = st.navigation(
    {"D&A": [
        st.Page(page_1,title = "Data description", icon ="⛵" ),
        st.Page(page_2,title = "Data analytics", icon ="📈" ),
    ],
     "AI":[
         st.Page(page_3, title = "Artificial Intelligence", icon = "🤖"),
     ]   
    }
)

pg.run()


