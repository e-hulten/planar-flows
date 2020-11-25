import os
import imageio


def make_gif_from_train_plots(fname: str) -> None:
    png_dir = "train_plots/"
    images = []
    sort = sorted(os.listdir(png_dir))
    for file_name in sort[1::1]:
        if file_name.endswith(".png"):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))

    imageio.mimsave("gifs/" + fname, images, duration=0.05)


if __name__ == "__main__":
    fname = "U_2.gif"  # U_1, U_2, U_3, U_4, ring
    make_gif_from_train_plots(fname)
