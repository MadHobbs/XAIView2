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
TEST_PATH = 'test'
LOC_PRED_PATH = 'xView2_first_place/pred34_loc'
LOC_SAL_PATH = 'saliency_maps/res34_loc_0_1_best_saliency'
DMG_PRED_PATH = 'xView2_first_place/res34cls2_0_tuned'
DMG_SAL_PATH = 'saliency_maps/res34_cls2_0_tuned_best_saliency'

def overlay(background, foreground, alpha):
    return Image.blend(background, foreground, alpha)

def display(show_truth, show_pred, show_saliency, raw, truth, pred, sal, alpha):
    if show_pred:
        if show_saliency:
            # show pred and saliency
            to_display = overlay(Image.fromarray((pred*255).astype('uint8')).convert('RGBA'), 
                                 sal.convert('RGBA'), alpha)
            if show_truth:
                # show truth, pred, and saliency
                to_display = overlay(Image.fromarray((truth*255).astype('uint8')).convert('RGBA'),
                                     to_display, alpha=0.5)
        else:
            if show_truth:
                # show truth and pred
                to_display = overlay(Image.fromarray((truth*255).astype('uint8')).convert('RGBA'), 
                                     Image.fromarray((pred*255).astype('uint8')).convert('RGBA'), alpha=0.5)
            else:
                # show pred
                to_display = pred
    elif show_saliency: 
        if show_truth:
            # show truth and saliency
            to_display = overlay(Image.fromarray((truth*255).astype('uint8')).convert('RGBA'),
                                 sal.convert('RGBA'), alpha)
        else:
            # show saliency (over raw)
            to_display =  overlay(raw.convert('RGBA'), sal.convert('RGBA'), alpha)
    elif show_truth:
        # show truth
        to_display = truth
    else:
        # show raw
        to_display = raw
    st.image(to_display)

def get_loc_mask(selected_scene):
    pre_image = cv2.imread(f"{TEST_PATH}/images/{selected_scene}_pre_disaster.png", cv2.IMREAD_UNCHANGED)
    pred = cv2.imread(f"{LOC_PRED_PATH}/{selected_scene}_pre_disaster_part1.png.png", cv2.IMREAD_UNCHANGED)
    loc = color.label2rgb(label=pred, image=color.rgba2rgb(pre_image), colors=['green'], bg_label=0, bg_color=None, kind='overlay')
    return loc

def get_dmg_mask(selected_scene):
    pre_path = f"{DMG_PRED_PATH}/{selected_scene}_pre_disaster_part1.png.png"
    msk1 = cv2.imread(pre_path, cv2.IMREAD_UNCHANGED)
    msk2 = cv2.imread(pre_path.replace('_part1', '_part2'), cv2.IMREAD_UNCHANGED)
    post_image = cv2.imread(f"{TEST_PATH}/images/{selected_scene}_post_disaster.png", cv2.IMREAD_UNCHANGED)
    msk = np.concatenate([msk1, msk2[..., 1:]], axis=2)
    pred=np.asarray(np.argmax(msk, axis=2), dtype=np.uint8)
    dmg = color.label2rgb(label=pred, image=color.rgba2rgb(post_image), colors=['green', 'yellow', 'orange', 'red'], 
                                            bg_label=0, bg_color=None, kind='overlay')
    return dmg

def get_loc_truth(selected_scene):
    pre_image = cv2.imread(f"{TEST_PATH}/images/{selected_scene}_pre_disaster.png", cv2.IMREAD_UNCHANGED)
    pre_truth = cv2.imread(f"{TEST_PATH}/targets/{selected_scene}_pre_disaster_target.png", cv2.IMREAD_UNCHANGED)
    loc = color.label2rgb(label=pre_truth, image=color.rgba2rgb(pre_image), colors=['green'], bg_label=0, bg_color=None, kind='overlay')
    return loc

def get_dmg_truth(selected_scene):
    post_image = cv2.imread(f"{TEST_PATH}/images/{selected_scene}_post_disaster.png", cv2.IMREAD_UNCHANGED)
    post_truth = cv2.imread(f"{TEST_PATH}/targets/{selected_scene}_post_disaster_target.png", cv2.IMREAD_UNCHANGED)
    dmg = color.label2rgb(label=post_truth, image=color.rgba2rgb(post_image), colors=['green', 'yellow', 'orange', 'red'], 
                                            bg_label=0, bg_color=None, kind='overlay')
    return dmg

#####################
### STREAMLIT APP ###
#####################

# title and layout
st.set_page_config(layout="wide")
st.title("xView2")

# widget layout
col0, col05, col1, col2 = st.beta_columns((1,1,6,6))

# # SessionState for prev and next buttons
ss = SessionState.get(file_index=0)

with col0:
    if st.button('Previous Scene'):
        ss.file_index -= 1
        #SessionState.get().file_index -= 1
        #SessionState.sync()
with col05:
    if st.button('Next Scene'):
        #SessionState.get().file_index += 1
        #SessionState.sync()
        ss.file_index += 1


# select scene (drop-down)
images = sorted(os.listdir(TEST_PATH+'/images'))
scenes = sorted(list(set([s.split('_')[0]+"_"+s.split('_')[1] for s in images])))
# images = os.walk(TEST_PATH+'/images').__next__()
# scenes = 
selected_scene = st.selectbox('Select Scene', scenes, index=ss.file_index)
ss.file_index = scenes.index(selected_scene)

# disaplay images
with col1:
    st.header('Pre-Disaster')
    # make buttons
    add_bldg_truth = st.checkbox("Show Building Truth")
    add_bldg_mask = st.checkbox("Show Building Prediction")
    add_bsal_map = st.checkbox("Show Building Saliency Map")
    # gather images
    pre = Image.open(f"{TEST_PATH}/images/{selected_scene}_pre_disaster.png")
    pre_truth = get_loc_truth(selected_scene)
    loc = get_loc_mask(selected_scene)
    bsal = Image.open(f"{LOC_SAL_PATH}/{selected_scene}_pre_disaster_loc_saliency_0.5.png")
    if add_bsal_map:
        alpha = st.slider("Transparency", min_value=0.0, max_value=1.0)
    else: alpha=None
    # show name of scene
    st.markdown("### "+selected_scene)
    # react to buttons and disply
    display(show_truth = add_bldg_truth, show_pred = add_bldg_mask, show_saliency = add_bsal_map, 
            raw=pre, truth=pre_truth, pred=loc, sal=bsal, alpha=alpha)
with col2:
    st.header('Post-Disaster')
    # make buttons
    add_dmg_truth = st.checkbox("Show Damage Truth")
    add_dmg_mask = st.checkbox("Show Damage Prediction")
    show_saliency = st.checkbox('Show Damage Saliency Map')
    # gather images
    post = Image.open(f"{TEST_PATH}/images/{selected_scene}_post_disaster.png")
    post_truth = get_dmg_truth(selected_scene)
    dmg = get_dmg_mask(selected_scene) 
    dsal_no_building = Image.open(f"{DMG_SAL_PATH}/{selected_scene}_pre_disaster_dmg_no-building_saliency_0.5.png")
    dsal_no_damage = Image.open(f"{DMG_SAL_PATH}/{selected_scene}_pre_disaster_dmg_no-damage_saliency_0.5.png")
    dsal_minor_damage = Image.open(f"{DMG_SAL_PATH}/{selected_scene}_pre_disaster_dmg_minor-damage_saliency_0.5.png")
    dsal_major_damage = Image.open(f"{DMG_SAL_PATH}/{selected_scene}_pre_disaster_dmg_major-damage_saliency_0.5.png")
    dsal_destroyed = Image.open(f"{DMG_SAL_PATH}/{selected_scene}_pre_disaster_dmg_destroyed_saliency_0.5.png")
    dsal_dict = {'No Building':dsal_no_building, 'No Damage':dsal_no_damage, 
                 'Minor Damage':dsal_minor_damage, 'Major Damage':dsal_major_damage, 
                 'Destroyed':dsal_destroyed}
    if show_saliency:
        alpha = st.slider("Transparency ", min_value=0.0, max_value=1.0)
        add_dsal_map = dsal_dict[st.selectbox("Damage Classes:", 
                                        options=['No Damage', 'Minor Damage', 
                                                'Major Damage', 'Destroyed'])]
    else: 
        add_dsal_map = None
        alpha = None
    # show name of scene
    st.markdown("### "+selected_scene)
    # react to buttons and display
    display(show_truth = add_dmg_truth, show_pred = add_dmg_mask, show_saliency = show_saliency, 
            raw=post, truth=post_truth, pred=dmg, sal=add_dsal_map, alpha=alpha)
    

