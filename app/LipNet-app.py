import sqlite3
from PIL import Image
import base64
import numpy as np
import os
import imageio
import tensorflow as tf
from Utils import load_data, num_to_char
from ModelUtils import load_model
import speech_recognition as sr
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Global variable for the database connection
conn = None

# Function to create user table if not exists
def create_user_table():
    global conn
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)')
    conn.commit()

# Function to close the database connection
def close_connection():
    global conn
    if conn:
        conn.close()

# Home Page
def home_page():
    # Background image URL (Replace with your image URL)
    url = "https://img.freepik.com/free-vector/technology-face-circuit-diagram-background_1017-18300.jpg"

    # Create a div element for the background image
    div_style = f"""
    <div style="background-image: url('{url}');
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: center;
                height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
    ">
    """

    # Overlay text on the background image with 100% width
    div_style += '<h1 style="color: dark blue; font-size: 40px; text-align: center; width: 100%;">LipNet - Video to Text Transcription</h1>'

    # Main Objective
    div_style += '<h2 style="color: teal; font-size: 24px; text-align: center; width: 100%;">Main Objective</h2>'
    div_style += '<p style="color: black; font-size: 18px; text-align: center; width: 100%;">LipNet is an end-to-end deep learning model designed to accurately perform lipreading by combining visual information from lip movements with language context. It recognizes and transcribes spoken words or phrases from video sequences.</p>'

    # Additional Information
    div_style += '<h2 style="color: teal; font-size: 24px; text-align: center; width: 100%;">Additional Information</h2>'
    div_style += '<p style="color: black; font-size: 18px; text-align: center; width: 100%;">LipNet uses Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Long Short-Term Memory (LSTM) cells to achieve high accuracy in lipreading and transcription tasks. This model finds applications in speech recognition and natural language processing (NLP).</p>'

    # Call-to-Action
    div_style += '<h2 style="color: teal; font-size: 24px; text-align: center; width: 100%;">Get Started</h2>'
    div_style += '<p style="color: black; font-size: 18px; text-align: center; width: 100%;">To get started, you can either login or register to access the LipNet app and start transcribing videos.</p>'

    div_style += "</div>"  # Close the div element

    # Render the background image and content
    st.markdown(div_style, unsafe_allow_html=True)

    # Footer
    st.markdown("- Developed by Krithan`.")
    st.markdown("---")

# LipNet Login and Registration System
def LipNet_login_registration_app():
    st.markdown("---")
    st.title("LipNet Login and Registration")
    st.markdown("---")

    # Dropdown box to select between "Login" and "Register"
    option = st.selectbox("Select an option:", ("Login", "Register"))

    if option == "Login":
        login_form()
    else:
        register_form()

# Login form
def login_form():
    username = st.text_input('Username', key='login_username')
    password = st.text_input('Password', type='password', key='login_password')

    # Validate if username and password are not empty
    if not username or not password:
        st.error('Please enter both username and password to login')
        return False

    if st.button('Login'):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
        result = c.fetchone()
        conn.close()

        if result:
            st.success('Login Successful!')
            st.success(f'Logged in as {username}')
            # User main page should be displayed here, customize as needed
            st.write("Welcome to the User main page!")
            st.session_state['logged_in'] = True  # Set logged_in session state to True
            return True
        else:
            st.error('Invalid Credentials')
            return False

# Registration form
def register_form():
    username = st.text_input('Username', key='register_username')

    # Validate the username
    if not username.isalpha() and username != "":
        st.error('Please enter only characters')
        username = ''  # Clear the input if it contains invalid characters

    password = st.text_input('Password', type='password', key='register_password')
    confirm_password = st.text_input('Confirm Password', type='password', key='confirm_password')

    if st.button('Register'):
        if not username or not password or not confirm_password:
            st.error('Please fill out all fields')
        elif password != confirm_password:
            st.error('Passwords do not match')
        else:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username=?', (username,))
            result = c.fetchone()

            if result:
                st.error('Username already exists')
            else:
                c.execute('INSERT INTO users VALUES (?, ?)', (username, password))
                conn.commit()
                conn.close()
                st.success('Registration successful')

# Prediction Page
def prediction_page():
    # SET UP THE SIDEBAR
    with st.sidebar:
        st.image('https://149695847.v2.pressablecdn.com/wp-content/uploads/2020/03/liopa_header_video_bg-1.jpg')
        st.markdown('---')
        st.title('LipNet')
        st.markdown('---')
        st.info('An application leveraging our video-to-text transcription model for precise spoken word conversion, enabling seamless video content analysis, search, and accessibility.')

    # Check if the user is logged in
    logged_in = st.session_state.get('logged_in', False)
    if not logged_in:
        st.error('You must be logged in to access the current page.')
        return

 
    # Add your prediction page content here
    st.markdown("---")
    st.title('LipNet Full Stack Application')
    st.markdown("---")

    # GENERATE A LIST OF VIDEO OPTIONS
    options = os.listdir(os.path.join('..', 'data', 's1'))
    st.markdown('---')
    selected_video = st.selectbox('Choose a Video', options)
    st.markdown('---')

    # GENERATE THE COLUMNS
    col1, col2 = st.columns(2)

    if options:
        with col1:
            st.info('The video below displays the converted video in mp4 format')
            file_path = os.path.join('..', 'data', 's1', selected_video)
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
            audio_file_path = 'converted_audio.wav'  # Output audio file path

            # RENDER THE MP4 VIDEO INSIDE A FRAME
            video = open('test_video.mp4', 'rb')
            video_bytes = video.read()
            st.markdown('---')
            st.markdown('**MP4 Video**')
            st.markdown('---')
            st.frame_width = 600
            st.frame_height = 450
            st.video(video_bytes)
            st.markdown('---')

        with col2:
            st.info('Visualization of the input video frames')
            video, annotations = load_data(tf.convert_to_tensor(file_path))
            video_array = tf.squeeze(video).numpy()
            fps = 10  # Increase the frame rate (fps) for a smoother and clearer animation
            frame_duration = int(1000 / fps)  # Calculate frame duration in milliseconds
            num_loops = 45  # Number of times to repeat the animation

            # Create a list to store repeated frames
            repeated_frames = []
            for _ in range(num_loops):
                repeated_frames.extend(video_array)

            repeated_frames = np.array(repeated_frames)  # Convert the list to a NumPy array
            imageio.mimsave('animation.gif', repeated_frames.astype('uint8'), duration=frame_duration)

            # LOAD AND DISPLAY THE ANIMATION.GIF INSIDE A FRAME
            with open('animation.gif', 'rb') as f:
                gif_bytes = f.read()
                st.markdown('---')
                st.markdown('**Animation GIF**')
                st.markdown('---')
                st.frame_width = 300
                st.frame_height = 450
                st.image(gif_bytes, width=600)
                st.markdown('---')
                st.write('The animation above displays a sequence of frames extracted from the input video. Each frame represents a snapshot of the video at a specific time. By analyzing these frames, the machine learning model can gain insights into the visual content and make predictions based on the observed patterns. This visualization helps us understand how the model perceives and processes the video data.')
                st.markdown('---')

        st.markdown('---')
        st.info('This is the output of the machine learning model as tokens')
        with st.spinner('Loading the model...'):
            model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

    # Predicted Output
   
    st.markdown('---')
    st.info('This is the Prediction of the LipNet model.')
    with st.spinner('Loading the model...'):
        model = load_model()
    yhat = model.predict(tf.expand_dims(video, axis=0))
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

    # Define the HTML and JavaScript code for the moving text effect
    moving_text_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .moving-text-container {{
                position: relative;
                background-color: #ffffff;
                padding: 40px;
                margin-top: 40px;
            }}

            .moving-text {{
                position: absolute;
                top: 0;
                left: 0;
                animation: moveText 10s infinite linear;
                white-space: nowrap;
                overflow: hidden;
                width: 100%;
            }}

            @keyframes moveText {{
                0% {{ transform: translateX(100%); }}
                100% {{ transform: translateX(-100%); }}
            }}
        </style>
    </head>
    <body>
        <div class="moving-text-container">
            <div class="moving-text" style="font-size: 40px; color: #ff5500;">{}</div>
        </div>
    </body>
    </html>
    """

    # Inject the converted prediction into the HTML code using format method
    moving_text_code = moving_text_code.format(converted_prediction)

    # Display the HTML code as a custom HTML component
    st.components.v1.html(moving_text_code)

    # Create a base64-encoded download link for the transcript
    transcript_base64 = base64.b64encode(converted_prediction.encode('utf-8')).decode('utf-8')
    href = f'<a href="data:text/plain;base64,{transcript_base64}" download="transcript.txt" class="custom-button">Download Transcript</a>'

    # Include CSS styling
    st.markdown(
        """
        <style>
        .custom-button {
            background-color: lavender;
            color: #ffffff;
            font-weight: bold;
            padding: 8px 16px;
            border-radius: 4px;
            border: none;
            cursor: pointer;
            margin-top: 10px;
        }

        .transcript-section {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the transcript section
    st.markdown('---')
    st.markdown('**Transcript**')
    st.markdown('---')

    # Show the transcript
    with st.expander('Show Transcript'):
        st.code(converted_prediction, language='text')

    # Show the download transcript link
    st.markdown('---')
    st.markdown('**Download Transcript**')
    st.markdown('---')
    st.markdown(href, unsafe_allow_html=True)


# About Page
def about_page():
    st.markdown("---")
    st.markdown("<h1 style='color: olive;'>About Page</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### How LipNet Works")
    st.markdown("LipNet is an end-to-end deep learning model designed for lipreading, which involves the task of converting lip movements into spoken words or phrases. It leverages a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to process and analyze video sequences of lip movements.")
    st.image("img1.jpg", caption="LipNet Architecture", use_column_width=True)
    st.markdown("The key components of LipNet include:")
    st.markdown("- **Spatial-Temporal CNN (STCNN):** The STCNN extracts spatiotemporal features from the lip image frames, enabling the model to capture both spatial patterns and temporal changes in lip movements over time.")
    st.markdown("- **Recurrent Neural Networks (RNNs):** The RNNs are used to process the extracted features over sequential frames, allowing LipNet to model the temporal dependencies and capture the context of the spoken words.")
    st.markdown("- **Connectionist Temporal Classification (CTC):** CTC is employed as the loss function to train LipNet. It enables the model to handle variable-length input and output sequences, making it suitable for lipreading, where the number of frames and spoken words may vary.")
    st.markdown("With this architecture, LipNet can accurately perform lipreading by combining visual information from lip movements with language context to recognize and transcribe spoken words or phrases.")
    
    st.markdown("### Performance Graph")
    st.markdown("The performance graph below shows the Word Error Rate (WER) for LipNet and several baseline models, evaluated on two splits: 'Unseen Speakers' and 'Overlapped Speakers'. WER measures the accuracy of lipreading by calculating the minimum number of word insertions, substitutions, and deletions required to transform the predicted transcript into the ground truth, divided by the number of words in the ground truth.")
    st.markdown("The lower the WER, the higher the accuracy of the lipreading model. As shown in the graph, LipNet outperforms the baseline models in both evaluation splits, demonstrating its effectiveness in lipreading tasks.")
    st.markdown("Note: The data in this demonstration is for illustrative purposes only.")

  
    # Performance data for your LipNet model and baseline models (replace with your actual WER values)
    models = ["LipNet", "Baseline Model 1", "Baseline Model 2", "Baseline Model 3"]
    wer_data = [10.5, 15.2, 18.7, 13.6]  # Replace these values with your actual WER values

    # Plot the performance graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(models, wer_data, color='teal')
    ax.set_title('Word Error Rate (WER) Comparison')
    ax.set_xlabel('Models')
    ax.set_ylabel('WER (%)')
    ax.grid(True)

     # Display the graph
    st.pyplot(fig)


    # Performance data for demonstration purposes
    models = ["LipNet", "Baseline-LSTM", "Baseline-2D", "Baseline-NoLM"]
    wer_data_word_level = [10.5, 23.1, 17.9, 12.3]  # Word Error Rate (WER) values for word-level prediction
    ser_data_sentence_level = [20.2, 35.8, 28.6, 21.7]  # Sentence Error Rate (SER) values for sentence-level prediction

    # Plot the comparison graph for both word-level and sentence-level predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(models))

    ax.bar(index, wer_data_word_level, bar_width, label='Word Level Prediction (WER)', color='teal')
    ax.bar([i + bar_width for i in index], ser_data_sentence_level, bar_width, label='Sentence Level Prediction (SER)', color='orange')

    ax.set_xlabel('Models')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Comparison between Word Level Prediction and Sentence Level Prediction')
    ax.set_xticks([i + bar_width/2 for i in index])
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True)

    # Display the graph
    st.pyplot(fig)

    st.markdown("### Applications of LipNet")
    st.markdown("LipNet has numerous applications in various domains:")
    st.markdown("- **Speech Recognition:** LipNet can be used in speech recognition systems to enhance the accuracy of transcriptions, especially in scenarios with noisy audio or when the speaker's voice is not clear.")
    st.markdown("- **Accessibility:** LipNet can improve accessibility for people with hearing impairments by converting spoken language to text in real-time.")
    st.markdown("- **Video Content Analysis:** LipNet can be applied in video content analysis to automatically extract spoken information from videos, making it easier to search and index video content based on the spoken words.")
    st.markdown("- **Surveillance and Security:** LipNet can assist in identifying and understanding spoken conversations in surveillance videos, contributing to security and investigation tasks.")
    st.markdown("- **Language Learning:** LipNet can aid language learners by providing transcriptions and translations of spoken language in videos or audio materials.")
    st.markdown("- **Human-Robot Interaction:** LipNet can be integrated into robotics systems to enable robots to understand and respond to spoken commands from humans.")
    st.markdown("These applications demonstrate the potential impact of LipNet in diverse fields, improving speech-related technologies and user experiences.")
    
  
    # Add more content to the About page as needed

# Main function to run the Streamlit app
def main():
    create_user_table()

    # SET PAGE CONFIGURATION
    st.set_page_config(layout="wide")

    # RENDER THE SIDEBAR
    st.sidebar.title('Navigation')
    options = ['Home', 'Login/Register', 'Prediction', 'About']
    selection = st.sidebar.selectbox('Go to', options)

    # RENDER THE SELECTED PAGE
    if selection == 'Home':
        home_page()
    elif selection == 'Login/Register':
        LipNet_login_registration_app()
    elif selection == 'Prediction':
        prediction_page()
    elif selection == 'About':
        about_page()

    close_connection()


if __name__ == '__main__':
    main()
