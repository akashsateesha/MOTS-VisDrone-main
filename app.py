import streamlit as st
from constants import get_tracking_algorithm_names, get_tracking_algorithm_value, DETECTION_MODELS, DETECTION_MODELS_PATH, CLASS_NAMES
from tempfile import NamedTemporaryFile
import track
from time import perf_counter

mainContainer = st.container()

if 'analysing' not in st.session_state:
    st.session_state['analysing'] = False

with mainContainer:
    st.title("Choose a video to analyze")
    file = st.file_uploader("Upload file", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False)
    
    if file is not None:
        st.video(file)
        detection_model = st.selectbox("Choose a Detection Model", DETECTION_MODELS)
        model_path = DETECTION_MODELS_PATH[detection_model]

        filter_classes = st.multiselect("Filter By Class", list(CLASS_NAMES.values()), default=list(CLASS_NAMES.values()))
        classes = [k for k, v in CLASS_NAMES.items() if v in filter_classes]
        
        st.info("Filtering by classes: {}".format(', '.join(filter_classes)))
        with_sahi = st.checkbox("With SAHI", value=False)
        if with_sahi:
            st.warning("SAHI might take a long time to process the video, so please be patient! 🕒💻🔍")
        tracking_algorithm_names = get_tracking_algorithm_names()
        tracking_algorithm = st.selectbox("Choose a tracking algorithm", tracking_algorithm_names)
        tracking_algorithm_value = get_tracking_algorithm_value(tracking_algorithm)
    
        if st.session_state['analysing'] == False:
            st.button("Analyze video", key="analyze_video", on_click=lambda: st.session_state.update({'analysing': True}), type="primary")
        else:
            with st.spinner("Analyzing your video with the powerful {} algorithm! 🚀✨ It might take a few minutes to work its magic, so sit back, relax, and enjoy the anticipation! 🕒💻🔍".format(tracking_algorithm)):
                with NamedTemporaryFile(dir='.', suffix=".mp4", prefix= file.name.split(".")[0]) as f:
                    f.write(file.getbuffer())
                    start_time = perf_counter()
                    result = track.main_streamlit(
                        source = f.name, 
                        tracking_method= tracking_algorithm_value,
                        model_path= model_path,
                        sahi = with_sahi,
                        classes= classes
                    )
                    end_time = perf_counter()
                    f.flush()
                    st.session_state.update({'analysing': False})
                    st.subheader("Analysed Video")
                    mm, ss = divmod(end_time - start_time, 60)
                    st.write("Video analysed in: {} minutes and {} seconds".format(int(mm), int(ss)))
                    st.video(result)
                
        