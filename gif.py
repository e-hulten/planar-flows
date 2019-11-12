import os
import imageio

target_distr = "U_3"  # U_1, U_2, U_3, U_4, ring

if not os.path.exists("gifs"):
    os.makedirs("gifs")

png_dir = "train_plots/"
images = []
sort = sorted(os.listdir(png_dir))
for file_name in sort[1::10]:
    if file_name.endswith(".png"):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))


imageio.mimsave("gifs/" + target_distr + ".gif", images, duration=0.025)

