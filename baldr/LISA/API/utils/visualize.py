import cv2
import numpy as np

def visualize(args, pred_masks, image_path, image_np):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        lv = len(hex_color)
        return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    colormap = ["#1f77b4","#d62728","#ff7f0e","#2ca02c","#9467bd","#8c564b","#e377c2",\
                "#7f7f7f","#bcbd22","#17becf","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf", \
                "#1f77b4","#41340e","#2ca02c","#b62728","#9467bd","#8ff64b","#e377c2","#7f7faa","#bcbd22","#17baaf"]
    colormap_rgb = [hex_to_rgb(color) for color in colormap]
    save_path_tot = "{}/{}_masked_img.jpg".format(
        args.vis_save_path, image_path.split("/")[-1].split(".")[0]
    )
    save_img = image_np.copy()
    
    for i, pred_mask in enumerate(pred_masks[0]):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()#[0]
        pred_mask = pred_mask > 0

        save_path = "{}/{}_mask_{}.jpg".format(
            args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
        )
        cv2.imwrite(save_path, pred_mask * 100)
        print("{} has been saved.".format(save_path))
        
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array(colormap_rgb[i % len(colormap_rgb)]) * 0.5
        )[pred_mask]
    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path_tot, save_img)
    print("{} has been saved.".format(save_path_tot))
    