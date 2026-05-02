import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base = "/Users/huseyin/Desktop/Python/AI/Resnet implementation/images"
img_top = mpimg.imread(f"{base}/depth_comparison.png")
img_bot = mpimg.imread(f"{base}/depth_comparison_clf.png")

h_top, w_top = img_top.shape[:2]
h_bot, w_bot = img_bot.shape[:2]
fig_w = 13
fig_h = fig_w * (h_top / w_top + h_bot / w_bot)

fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h))
for ax, img in zip(axes, [img_top, img_bot]):
    ax.imshow(img)
    ax.axis("off")

plt.subplots_adjust(hspace=0.05, left=0, right=1, top=1, bottom=0)
plt.savefig(f"{base}/depth_comparison_combined.png", dpi=200, bbox_inches="tight", transparent=True)
plt.show()
