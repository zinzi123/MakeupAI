import streamlit as st
import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from collections import Counter

nltk.download('punkt')
nltk.download('wordnet')

# Load environment variables
load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Initialize search tool
search_tool = SerperDevTool()

def create_llm(use_gpt=True):
    """Create the LLM model."""
    if use_gpt:
        return ChatOpenAI(model="gpt-4o-mini")
    else:
        return Ollama(model="llama3.2")

def create_agents(user_preferences, llm):
    """Create agents for the beauty box tasks."""
    preference_analyzer = Agent(
        role="Preference Analyzer",
        goal="Analyze user preferences and requirements to personalize the beauty box.",
        backstory="You are skilled in understanding user needs and mapping them to suitable products.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )
    trend_analyzer = Agent(
        role="Trend Analyzer",
        goal="Identify current beauty trends relevant to the user's preferences.",
        backstory="You specialize in trend analysis for the beauty industry.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )
    item_selector = Agent(
        role="Item Selector",
        goal="Curate beauty items for the personalized box based on preferences and trends.",
        backstory="You are an expert in selecting beauty products that align with user preferences.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )
    feedback_analyzer = Agent(
        role="Feedback Analyzer",
        goal="Analyze user feedback on previous beauty boxes to improve future recommendations.",
        backstory="You excel in extracting insights from user feedback to enhance personalization.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )
    return [preference_analyzer, trend_analyzer, item_selector, feedback_analyzer]

def create_tasks(user_preferences, agents):
    """Create tasks for the beauty box process."""
    preference_task = Task(
        description="Analyze user preferences and requirements to guide product selection.",
        agent=agents[0],
        expected_output="A detailed summary of user preferences, including skin type, allergies, and product types."
    )
    trend_task = Task(
        description="Identify trending beauty products relevant to user preferences.",
        agent=agents[1],
        expected_output="A list of trending products and categories relevant to the user's profile."
    )
    selection_task = Task(
        description="Select beauty items for the customized box based on preferences and trends.",
        agent=agents[2],
        expected_output="A curated list of beauty products for the user."
    )
    feedback_task = Task(
        description="Analyze feedback from previous beauty boxes to refine future selections.",
        agent=agents[3],
        expected_output="Insights and recommendations for improving future beauty boxes."
    )
    return [preference_task, trend_task, selection_task, feedback_task]

def generate_wordcloud(text):
    """Generate a word cloud from text."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt

def run_beauty_box_process(user_preferences, use_gpt=True, max_retries=3):
    """Run the beauty box customization process."""
    llm = create_llm(use_gpt)
    agents = create_agents(user_preferences, llm)
    tasks = create_tasks(user_preferences, agents)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )

    for attempt in range(max_retries):
        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                st.info("Retrying...")
                time.sleep(5)
            else:
                st.error("Max retries reached. Unable to complete the task.")
                return None

# Streamlit UI
st.title("Personalized Beauty Box Customization")

st.sidebar.header("User Preferences")
skin_type = st.sidebar.selectbox("Skin Type", options=["Dry", "Oily", "Combination", "Sensitive"])
allergies = st.sidebar.text_input("Allergies (if any)")
favorite_brands = st.sidebar.text_input("Favorite Brands")
preferred_products = st.sidebar.text_area("Preferred Product Types (e.g., moisturizer, lipstick)")
use_gpt = st.sidebar.radio("Choose the LLM model", options=["GPT", "Llama"], index=0) == "GPT"

if st.button("Generate Beauty Box"):
    user_preferences = {
        "skin_type": skin_type,
        "allergies": allergies,
        "favorite_brands": favorite_brands,
        "preferred_products": preferred_products
    }

    with st.spinner("Customizing your beauty box..."):
        result = run_beauty_box_process(user_preferences, use_gpt)

    if result:
        st.success("Beauty Box Customized Successfully!")

        # Assuming the text is inside result.output or a similar attribute
        text_output = getattr(result, "output", str(result))  # Fallback to string representation if 'output' doesn't exist

        st.text_area("Report", value=text_output, height=300)

        # Generate word cloud
        st.subheader("Word Cloud of Trends")
        wordcloud_plot = generate_wordcloud(text_output)
        st.pyplot(wordcloud_plot)
    else:
        st.error("Failed to customize the beauty box.")
