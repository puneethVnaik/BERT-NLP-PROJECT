import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import io

# Set page configuration
st.set_page_config(
    page_title="Contextual Language Understanding with Transformer Models",
    page_icon="📊",
    layout="wide"
)

# Main app
def main():
    # App title and description
    st.title("Contextual Language Understanding with Transformer Models")
    st.subheader("BERT-based Sentiment Analysis")
    
    st.write("""
    This application uses a BERT transformer model to analyze the sentiment of text.
    The model classifies text as either positive or negative based on contextual understanding.
    """)
    
    # Create tabs for different types of analysis
    analysis_type = st.radio(
        "Select Analysis Type:",
        ["Basic Sentiment Analysis", "Batch Processing", "Real-time Analysis"]
    )
    
    # API endpoint for the deployed model
    API_ENDPOINT = "http://localhost:8000/predict"  # Replace with your actual API endpoint when deployed
    
    if analysis_type == "Basic Sentiment Analysis":
        st.write("### Enter text for sentiment analysis")
        user_input = st.text_area("", "I absolutely loved this product! It exceeded all my expectations.", height=150)
        
        if st.button("Analyze Sentiment"):
            # Add a spinner while processing
            with st.spinner("Analyzing sentiment..."):
                try:
                    # In a real implementation, this would call the API
                    # response = requests.post(API_ENDPOINT, json={"text": user_input})
                    # result = response.json()
                    
                    # For demo purposes, simulate API call
                    time.sleep(1)
                    
                    # Simple rule-based sentiment simulation
                    positive_words = ["love", "great", "excellent", "best", "fantastic", "wonderful", "enjoyed", "amazing", "exceeded"]
                    negative_words = ["hate", "worst", "terrible", "bad", "awful", "disappointing", "boring", "waste"]
                    
                    pos_count = sum(1 for word in positive_words if word in user_input.lower())
                    neg_count = sum(1 for word in negative_words if word in user_input.lower())
                    
                    sentiment = "positive" if pos_count > neg_count else "negative"
                    confidence = min(0.5 + 0.1 * abs(pos_count - neg_count), 0.99)
                    
                    # Display result with visualizations
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if sentiment == "positive":
                            st.success(f"Sentiment: {sentiment.upper()}")
                        else:
                            st.error(f"Sentiment: {sentiment.upper()}")
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    with col2:
                        # Create sentiment visualization
                        fig, ax = plt.subplots(figsize=(4, 0.8))
                        
                        # Define colors based on sentiment
                        if sentiment == "positive":
                            colors = ["#ff9999", "#ffcc99", "#99ff99"]
                        else:
                            colors = ["#99ff99", "#ffcc99", "#ff9999"]
                            
                        # Create gauge chart
                        ax.barh([0], [2], left=[-1], color=colors)
                        ax.barh([0], [0.04], left=[confidence*2-1-0.02], color='black')
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(-1, 1)
                        ax.axis('off')
                        ax.set_title(f"Sentiment Score: {confidence:.2f}")
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {str(e)}")
    
    elif analysis_type == "Batch Processing":
        st.write("### Upload a CSV file with a column named 'text' containing the texts to analyze")
        
        # Provide sample CSV for download
        sample_data = {
            'text': [
                "This product is amazing, I love it!",
                "The service was terrible and the staff was rude.",
                "Pretty average experience, nothing special.",
                "I highly recommend this to everyone!",
                "Complete waste of money, don't buy it."
            ]
        }
        sample_df = pd.DataFrame(sample_data)
        
        # Create a download button for the sample CSV
        buffer = io.StringIO()
        sample_df.to_csv(buffer, index=False)
        st.download_button(
            label="Download Sample CSV",
            data=buffer.getvalue(),
            file_name="sample_texts.csv",
            mime="text/csv"
        )
        
        # File uploader for CSV
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Load the CSV
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("CSV must contain a column named 'text'")
            else:
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("Process Batch"):
                    with st.spinner("Analyzing batch of texts..."):
                        # In a real implementation, we would call the API for each text
                        # Process each text (simulated for demo)
                        results = []
                        confidence_scores = []
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(df['text']):
                            # Simple rule-based sentiment for demo
                            positive_words = ["love", "great", "excellent", "best", "fantastic", "amazing", "recommend"]
                            negative_words = ["hate", "worst", "terrible", "bad", "awful", "waste", "rude"]
                            
                            # Ensure text is treated as string
                            text_str = str(text)
                            
                            pos_count = sum(1 for word in positive_words if word in text_str.lower())
                            neg_count = sum(1 for word in negative_words if word in text_str.lower())
                            
                            # Calculate sentiment and confidence
                            sentiment = "positive" if pos_count > neg_count else "negative"
                            confidence = min(0.5 + 0.1 * abs(pos_count - neg_count), 0.99)
                            
                            results.append(sentiment)
                            confidence_scores.append(confidence)
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(df))
                            time.sleep(0.05)  # Simulate processing time
                        
                        # Add results to dataframe
                        df['sentiment'] = results
                        df['confidence'] = confidence_scores
                        
                        # Display results
                        st.success(f"Processed {len(df)} texts")
                        st.write(df)
                        
                        # Create a chart of sentiment distribution
                        st.subheader("Sentiment Distribution")
                        sentiment_counts = df['sentiment'].value_counts()
                        fig, ax = plt.subplots(figsize=(8, 5))
                        colors = ['#ff9999' if x == 'negative' else '#99ff99' for x in sentiment_counts.index]
                        sentiment_counts.plot(kind='bar', ax=ax, color=colors)
                        plt.title("Sentiment Distribution")
                        plt.ylabel("Count")
                        plt.xlabel("Sentiment")
                        st.pyplot(fig)
                        
                        # Display confidence distribution
                        st.subheader("Confidence Distribution")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.histplot(df['confidence'], bins=10, kde=True, ax=ax)
                        plt.title("Confidence Score Distribution")
                        plt.xlabel("Confidence Score")
                        plt.ylabel("Count")
                        st.pyplot(fig)
                        
                        # Provide download option for results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv",
                        )
    
    elif analysis_type == "Real-time Analysis":
        st.write("### Real-time Sentiment Analysis Simulation")
        st.write("This feature simulates real-time sentiment analysis of streaming text data.")
        
        # Simulate a stream of texts
        if st.button("Start Real-time Analysis"):
            # Create a placeholder for the line chart
            chart_placeholder = st.empty()
            text_placeholder = st.empty()
            
            # Create some sample texts with timestamps
            sample_texts = [
                "I love this product! It's amazing.",
                "Not satisfied with the quality.",
                "Average experience, nothing special.",
                "Terrible customer service.",
                "Exceeded my expectations, very happy!",
                "Product broke after one use.",
                "Great value for money.",
                "Would recommend to friends.",
                "Disappointing performance overall.",
                "Best purchase I've made this year!"
            ]
            
            # Create DataFrame to store results
            results_df = pd.DataFrame(columns=['Time', 'Text', 'Sentiment', 'Score'])
            
            # Initialize the chart
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Simulate real-time processing
            sentiment_scores = []
            timestamps = []
            
            for i in range(10):
                # Get current time
                current_time = pd.Timestamp.now().strftime('%H:%M:%S')
                timestamps.append(current_time)
                
                # Get a sample text
                text = sample_texts[i]
                text_placeholder.text(f"Processing: \"{text}\"")
                
                # Simulate sentiment analysis
                positive_words = ["love", "great", "excellent", "best", "fantastic", "happy", "amazing", "recommend"]
                negative_words = ["hate", "worst", "terrible", "bad", "awful", "disappointing", "broke"]
                
                pos_count = sum(1 for word in positive_words if word in text.lower())
                neg_count = sum(1 for word in negative_words if word in text.lower())
                
                # Calculate sentiment score (-1 to 1)
                if pos_count == neg_count:
                    score = 0
                else:
                    score = (pos_count - neg_count) / max(1, pos_count + neg_count)
                
                sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
                sentiment_scores.append(score)
                
                # Add to DataFrame
                new_row = pd.DataFrame({
                    'Time': [current_time],
                    'Text': [text],
                    'Sentiment': [sentiment],
                    'Score': [score]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                # Update chart
                ax.clear()
                ax.plot(range(len(sentiment_scores)), sentiment_scores, marker='o', linestyle='-', color='blue')
                ax.set_ylim(-1.1, 1.1)
                ax.set_xlim(-0.5, max(9.5, len(sentiment_scores) - 0.5))
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                ax.set_xlabel('Text Index')
                ax.set_ylabel('Sentiment Score')
                ax.set_title('Real-time Sentiment Analysis')
                ax.grid(True, alpha=0.3)
                
                # Add labels for each point
                for j, (s, txt) in enumerate(zip(sentiment_scores, sample_texts[:len(sentiment_scores)])):
                    ax.annotate(f"{j+1}", (j, s), xytext=(0, 10 if s > 0 else -15),
                                textcoords='offset points', ha='center')
                
                # Update the chart display
                chart_placeholder.pyplot(fig)
                
                # Simulate processing delay
                time.sleep(1)
            
            # Clear the processing text placeholder
            text_placeholder.empty()
            
            # Show final results
            st.success("Real-time analysis completed")
            
            # Display results table
            st.subheader("Analysis Results")
            st.dataframe(results_df)
            
            # Show final statistics
            st.subheader("Analysis Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                positive_count = sum(1 for score in sentiment_scores if score > 0)
                st.metric("Positive Texts", positive_count)
            with col2:
                negative_count = sum(1 for score in sentiment_scores if score < 0)
                st.metric("Negative Texts", negative_count)
            with col3:
                neutral_count = sum(1 for score in sentiment_scores if score == 0)
                st.metric("Neutral Texts", neutral_count)

# Footer with additional information
def footer():
    st.markdown("---")
    st.markdown("""
    **About this application:**  
    This sentiment analysis tool is part of the "Contextual Language Understanding with Transformer Models" project,
    which aims to enhance NLP capabilities across various tasks such as sentiment analysis, 
    text summarization, and question answering.
    """)
    
    # Team information in the footer
    with st.expander("Project Team"):
        team_data = {
            "Name": ["M Vikram", "Preethesh", "Puneeth V", "Shashanka B S"],
            "CAN ID": ["CAN_33321089", "CAN_33320059", "CAN_33322066", "CAN_33322585"]
        }
        st.table(pd.DataFrame(team_data))

# Run the application
if __name__ == "__main__":
    main()
    footer()