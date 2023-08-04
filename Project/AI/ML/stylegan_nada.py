import os

pretrained_model_dir = os.path.join("/content", "models")
os.makedirs(pretrained_model_dir, exist_ok=True)

restyle_dir = os.path.join("/content", "restyle")
stylegan_ada_dir = os.path.join("/content", "stylegan_ada")
stylegan_nada_dir = os.path.join("/content", "stylegan_nada")

output_dir = os.path.join("/content", "output")

output_model_dir = os.path.join(output_dir, "models")
output_image_dir = os.path.join(output_dir, "images")

