import streamlit as st
import SessionState
import matplotlib.pyplot as plt 
import os

# set global variables
PATH_TO_DATA = 'test'

# title and layout
st.set_page_config(layout="wide")
st.markdown("# xView2")

# widget layout
prev_img, next_img, col1, col2 = st.beta_columns((1,1,6,6))

# # SessionState for prev and next buttons
ss = SessionState.get(file_index=0)
if prev_img.button('Previous Scene'):
    ss.file_index -= 1
if next_img.button('Next Scene'):
    ss.file_index += 1

# select location (drop-down)
images = os.listdir(f'{PATH_TO_DATA}/images')
locations = list(set([s.split('_')[0]+"_"+s.split('_')[1] for s in images]))
selected_loc = prev_img.selectbox('Select Location', locations, index=ss.file_index)

# view images
pre_path = f"{PATH_TO_DATA}/images/{selected_loc}_pre_disaster.png"
post_path = f"{PATH_TO_DATA}/images/{selected_loc}_post_disaster.png" 
with col1:
    st.header('Pre-Disaster')
    st.image(pre_path)
with col2:
    st.header('Post-Disaster')
    st.image(post_path)
with col1:
    st.header('Localization Mask')
with col2:
    st.header('Damage Mask')
with col1:
    st.header('Localization Saliency Map')
with col2:
    st.header('Damage Saliency Map')
