import streamlit as st
import SessionState
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np
import os
from PIL import Image
from skimage import color
import cv2

# set global variables
TEST_PATH = 'test/images'
LOC_PRED_PATH = 'xView2_first_place/pred34_loc'
LOC_SAL_PATH = 'saliency_maps/res34_loc_0_1_best_saliency'
DMG_PRED_PATH = 'xView2_first_place/res34cls2_0_tuned'
DMG_SAL_PATH = 'saliency_maps/res34_cls2_0_tuned_best_saliency'

# helpers
def load_and_convert(fpath):
    im = Image.open(fpath)
    return im.convert('RGBA')

def overlay(background, foreground, alpha=0.5):
    return Image.blend(background, foreground, alpha)

def display(show_mask, show_saliency, raw, mask, sal):
    if show_mask:
        if show_saliency:
            to_display = overlay(mask, sal)
        else: 
            to_display = mask
    elif show_saliency: 
        to_display = overlay(raw, sal)
    else:
        to_display = raw
    st.image(to_display)
# def display(show_mask, show_saliency, raw, mask, sal):
#     if show_mask:
#         raw_n_mask= overlay(raw, mask)
#         if show_saliency:
#             to_display = overlay(raw_n_mask, sal)
#         else: 
#             to_display = raw_n_mask
#     elif show_saliency: 
#         to_display = overlay(raw, sal)
#     else:
#         to_display = raw
#     st.image(to_display)

def get_loc_mask(selected_scene):
    pre_image = cv2.imread(f"{TEST_PATH}/{selected_scene}_pre_disaster.png", cv2.IMREAD_UNCHANGED)
    pred = cv2.imread(f"{LOC_PRED_PATH}/{selected_scene}_pre_disaster_part1.png.png", cv2.IMREAD_UNCHANGED)
    loc = color.label2rgb(pred, pre_image, colors=['black', 'green'], bg_color='black')
    return loc

def get_dmg_mask(selected_scene):
    pre_path = f"{DMG_PRED_PATH}/{selected_scene}_pre_disaster_part1.png.png"
    msk1 = cv2.imread(pre_path, cv2.IMREAD_UNCHANGED)
    msk2 = cv2.imread(pre_path.replace('_part1', '_part2'), cv2.IMREAD_UNCHANGED)
    post_image = cv2.imread(f"{TEST_PATH}/{selected_scene}_post_disaster.png", cv2.IMREAD_UNCHANGED)
    msk = np.concatenate([msk1, msk2[..., 1:]], axis=2)
    pred=np.asarray(np.argmax(msk, axis=2), dtype=np.uint8)
    dmg = color.label2rgb(pred, post_image, colors=['black', 'green', 'yellow', 'orange', 'red'], bg_color='black')
    return dmg

#####################
### STREAMLIT APP ###
#####################

# title and layout
st.set_page_config(layout="wide")
st.title("xView2")

# widget layout
prev_img, next_img, col1, col2 = st.beta_columns((1,1,6,6))

# # SessionState for prev and next buttons
ss = SessionState.get(file_index=0)
if prev_img.button('Previous Scene'):
    ss.file_index -= 1
if next_img.button('Next Scene'):
    ss.file_index += 1

# select scene (drop-down)
images = sorted(os.listdir(TEST_PATH))
scenes = sorted(list(set([s.split('_')[0]+"_"+s.split('_')[1] for s in images])))
selected_scene = prev_img.selectbox('Select Scene', scenes, index=ss.file_index)

# disaplay images
with col1:
    st.header('Pre-Disaster')
    # make buttons
    add_bldg_mask = st.checkbox("Show Building Prediction")
    #add_bldg_truth = st.checkbox("Show Damage Truth")
    add_bsal_map = st.checkbox("Show Building Saliency Map")
    # gather images
    pre = load_and_convert(f"{TEST_PATH}/{selected_scene}_pre_disaster.png")
    loc = get_loc_mask(selected_scene)
    bsal = load_and_convert(f"{LOC_SAL_PATH}/{selected_scene}_pre_disaster_loc_saliency_0.5.png")
    # show name of scene
    st.markdown("### "+selected_scene)
    # react to buttons and disply
    display(show_mask = add_bldg_mask, show_saliency = add_bsal_map, 
            raw=pre, mask=loc, sal=bsal)
with col2:
    st.header('Post-Disaster')
    # make buttons
    add_dmg_mask = st.checkbox("Show Damage Prediction")
    #add_dmg_truth = st.checkbox("Show Damage Truth")
    add_dsal_map = st.checkbox("Show Damage Saliency Map")
    # gather images
    post = load_and_convert(f"{TEST_PATH}/{selected_scene}_post_disaster.png")
    dmg = get_dmg_mask(selected_scene) #load_and_convert(f"{DMG_PRED_PATH}/{selected_scene}_pre_disaster_part1.png.png")
    dsal_no_building = load_and_convert(f"{DMG_SAL_PATH}/{selected_scene}_pre_disaster_dmg_no-building_saliency_0.5.png")
    dsal_no_damage = load_and_convert(f"{DMG_SAL_PATH}/{selected_scene}_pre_disaster_dmg_no-damage_saliency_0.5.png")
    dsal_minor_damage = load_and_convert(f"{DMG_SAL_PATH}/{selected_scene}_pre_disaster_dmg_minor-damage_saliency_0.5.png")
    dsal_major_damage = load_and_convert(f"{DMG_SAL_PATH}/{selected_scene}_pre_disaster_dmg_major-damage_saliency_0.5.png")
    dsal_destroyed = load_and_convert(f"{DMG_SAL_PATH}/{selected_scene}_pre_disaster_dmg_destroyed_saliency_0.5.png")
    # show name of scene
    st.markdown("### "+selected_scene)
    # react to buttons and display
    display(show_mask = add_dmg_mask, show_saliency = add_dsal_map, 
            raw=post, mask=dmg, sal=dsal_no_building)
    

