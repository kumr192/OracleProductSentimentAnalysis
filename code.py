import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Load the data
data = pd.read_csv("brand_perception_sample_data_with_products.csv")

# Enter OpenAI API key at runtime
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

# Initialize Chat Model if API key is provided
if openai_api_key:
    # Define the chat model with the API key
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    # Define Streamlit UI
    st.title("Product/Brand Sentiment Analyzer")
    st.write("Analyzes customer sentiment and suggests interventions to improve brand perception.")

    # Filter data by Product and Product Feature
    product = st.selectbox("Select Product", data["Product"].unique())
    feature = st.selectbox("Select Product Feature", data["ProductFeature"].unique())
    sentiment_category = st.selectbox("Select Sentiment Category", ["Positive", "Negative", "Neutral"])

    # Filter data based on selections
    filtered_data = data[(data["Product"] == product) &
                         (data["ProductFeature"] == feature) &
                         (data["SentimentCategory"] == sentiment_category)]

    # Display summary of filtered data
    st.write(f"Selected product: {product} with {sentiment_category} sentiment for {feature}")

    # Analyze common issues or positive aspects
    common_items = filtered_data["CommonIssues"].value_counts().reset_index()
    common_items.columns = ["Item", "Count"]

    if sentiment_category == "Positive":
        st.write("Common Positive Aspects Identified:")
    else:
        st.write("Common Issues Identified:")

    st.table(common_items)

    # Optionally, display pie chart of top 10 issues
    if st.checkbox("Show detailed data"):
        # Display pie chart of top 10 issues
        top_10_issues = common_items.head(10)
        fig, ax = plt.subplots()
        ax.pie(top_10_issues["Count"], labels=top_10_issues["Item"], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

    # Generate targeted recommendations based on sentiment category
    if st.button("Generate Recommendations"):
        if sentiment_category in ["Negative", "Neutral"]:
            input_prompt = HumanMessage(content=f"""
                Based on the analysis, we have identified a pattern of {sentiment_category} sentiment regarding {feature} in {product}.
                Customers have reported issues such as {", ".join(common_items["Item"].tolist()[:3])}. Suggest targeted interventions to improve customer satisfaction.
            """)
        elif sentiment_category == "Positive":
            input_prompt = HumanMessage(content=f"""
                {product} has received positive feedback for {feature}. Customers are happy, but we want to further improve.
                What recommendations can keep customers even more satisfied and engaged?
            """)

        # Call the chat model with the generated message
        response = chat_model([input_prompt])
        st.write("### Recommendations:")
        st.write(response.content)
else:
    st.warning("Please enter your OpenAI API key to proceed.")
