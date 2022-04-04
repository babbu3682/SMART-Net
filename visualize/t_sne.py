# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE





# import torch.nn.functional as F

# features   = {}
# def get_features(name):
#     def hook(model, input, output):
#         features[name] = output.detach()
#     return hook

# device = 'cuda'
# x_true = list()
# y_true = list()
# y_pred = list()

# with torch.no_grad():
#     for test_data in tqdm(test_loader):
#         model.avgpool.register_forward_hook(get_features('feat'))    
        
#         test_images = test_data[0].to(device)
#         test_labels = test_data[1].to(device)
        
#         pred_logit = model(test_images)
        
#         x_true.append(cv2.resize(test_images.squeeze().detach().cpu().numpy(), (64,64), interpolation = cv2.INTER_AREA))
#         y_true.append(test_labels[0].detach().cpu().numpy())
#         y_pred.append(features['feat'].detach().cpu().numpy())
        

# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# feat_input = np.stack(y_pred, axis=0)
# label_list = np.stack(y_true, axis=0)

# embed  = TSNE(n_components=2, random_state=42).fit_transform(feat_input.squeeze())

# import seaborn as sns
# import matplotlib.patheffects as PathEffects

# num_classes = 68
# palette = np.array(sns.color_palette("hls", num_classes))
# label_map = np.array([0,1,2,3,4,5,6])
# # create a scatter plot.
# fig = plt.figure(figsize=(50, 50))
# plt.scatter(embed[:, 0], embed[:, 1], lw=0, s=20, c=palette[label_list.astype(np.int)])

# plt.xlim(-25, 25)
# plt.ylim(-25, 25)
# plt.axis('off')
# plt.axis('tight')

# class_label_list = ['Chest_Decubitus', 'Abdomen_Lateral', 'Mastoid', 'Knee_Lateral',
# 'Upper_Extremity', 'Toe', 'Foot_Calcaneus', 'Chest_Rib', 'Finger',
# 'Leg_Oblique', 'Foot_AP', 'Pevis_coccyx_sacrum', 'Orbit', 'Ankle_AP',
# 'Zygomatic', 'Foot_Hindfoot', 'Pelvis_Oblique', 'Foot_Oblique',
# 'Whole_Lower_AP', 'Abdomen_KUB', 'Chest_frontal', 'Shoulder_AP',
# 'Knee_Oblique', 'Foot_Lateral', 'T_L_Spine', 'Forearm_Oblique',
# 'Abdomen_upright', 'Hand_Oblique', 'T_Spine_AP', 'Wrist_AP',
# 'Chest_Lateral', 'Pelvis_Lateral', 'Chest_Clavicle', 'Nose_Lateral',
# 'Pelvis_Frogleg', 'Ankle_Lateral', 'Ankle_Stress', 'Skull_Towne',
# 'Shoulder_Axial', 'Pelvis_SI_joint', 'Pelvis_Translateral',
# 'Ankle_Mortise', 'Hand_PA', 'L_Spine_Lateral', 'Hand_Lateral',
# 'Wrist_Oblique', 'Knee_AP', 'Skull_Lateral', 'Knee_Skyline',
# 'C_spine_Lateral', 'Knee_Stress', 'Leg_AP', 'Pelvis_AP',
# 'Humerus_Oblique', 'Femur_AP', 'Abdomen_supine', 'Whole_Spine_AP',
# 'C_spine_Atlas', 'Cochlea', 'Mandible', 'Skull_AP', 'Nose_PNS',
# 'L_Spine_Oblique', 'L_Spine_AP', 'Elbow_Lateral', 'Whole_Lower_Lateral',
# 'Skull_Tangential', 'C_spine_AP']

# plt.title('T-SNE', fontsize=20)
# plt.axis('off')

# # We add the labels for each digit.
# txts = []
# for i in range(num_classes):
#     # Position of each label.
#     xtext, ytext = np.median(embed[label_list == i, :], axis=0)
#     plt.text(xtext, ytext, str(i), fontsize=24)
#     plt.scatter([], [], c=np.array([palette[i]]), marker='o', label='['+str(i)+']'+str(class_label_list[i]))
#     fig.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
    
# plt.legend(loc='lower left', fontsize=15)
# plt.show()