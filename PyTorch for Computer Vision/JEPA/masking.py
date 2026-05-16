import math
import torch

class IJEPAMaskSampler:
    """
    Follows the masking section in A.1 in the IJEPA paper
    """
    def __init__(
        self,
        img_size = 224,
        patch_size = 16,
        enc_mask_scale = (0.85, 1.0),  # num patch scales of context
        pred_mask_scale = (0.15, 0.2), # num patch scales of targets
        aspect_ratio = (0.75, 1.5), # aspect ratio range of targets
        nenc = 1,     # number of context blocks we want
        npred = 4,    # number of target blocks we want
        min_keep = 4, # smallest number of patches to be considered valid
        allow_overlap = False,
    ):
        
        self.height = self.width = img_size // patch_size
        self.num_patches = self.height * self.width
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
    
    def _sample_block_size(
        self,
        scale, 
        aspect_ratio_scale
    ):
        
        """sample (h,w) for one block"""
        
        rand = torch.rand(1).item() # [0,1]

        min_s, max_s = scale
        mask_scale = min_s + rand * (max_s - min_s) # some val between min_s,max_s
        max_keep = int(self.height * self.width * mask_scale) # num patches to keep

        min_ar, max_ar = aspect_ratio_scale 
        aspect_ratio = min_ar + rand * (max_ar - min_ar) # some val between min_ar, max_ar

        ### get (h,w) that has max_keep elements inside w/ the wanted aspect ratio
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
 
        # Clamp to grid
        while h >= self.height: h -= 1
        while w >= self.width:  w -= 1
 
        return h, w
    
    def _sample_block_mask(self, b_size, acceptable_regions=None):
        """
        Sample one rectangular block of shape (h, w).
        If acceptable_regions is given, multiply the candidate mask by each
        complement mask to restrict placement and progressively relaxed on
        timeout
        """

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        h, w = b_size

        while not valid_mask:

            ### sample the top left corner of the block
            top = torch.randint(0, self.height - h, (1,)).item()
            left = torch.randint(0, self.width  - w, (1,)).item()

            ### create mask and populate with 1 for selected
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1

            ### acceptable regions is a list of masks that acts as a filter
            ### this will be used to ensure that our encoder (context) blocks
            ### from overlapping with the predictor (target) blocks
            ### the issue is that because this is random, this filtering may
            ### cause us to produce a tiny mask that doesnt have atleast min_keep
            ### locations. So we will gradually relax the constraint with every
            ### failure to grab a mask!
            if acceptable_regions is not None:
                N = max(int(len(acceptable_regions) - tries), 0)
                for k in range(N):
                    mask *= acceptable_regions[k]
            
            mask_ids = torch.nonzero(mask.flatten()).squeeze(1)
            valid_mask = len(mask_ids) > self.min_keep

            if not valid_mask:
                timeout -= 1 # we get 20 tries to get the mask right
                if timeout == 0: # if we cant in 20, then relax constraint and try again
                    tries += 1
                    timeout = og_timeout

        ### Get the inverse of the mask (all non-selected portions)
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0

        return mask_ids, mask_complement

    def sample(self, batch_size):
        
        ### sample block size. The e_size is for our context. We will grab a bunch
        ### of small masks for our targets (0.15 to 0.2). Then we will grab a large
        ### chunk of the image as context, and remove any overlap with our sampled 
        ### target locations as we dont really want overlap (allow_overlap=false)
        p_size = self._sample_block_size(self.pred_mask_scale, self.aspect_ratio)
        e_size = self._sample_block_size(self.enc_mask_scale,  (1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        
        ### different samples in the batch will have different number of prediction
        ### patches selected, or different number of context patches selected. 
        ### We could do padding, and mask out the invalid positions, but the original
        ### IJEPA implementation just keeps the minimum available across all samples
        min_keep_pred = self.num_patches
        min_keep_enc = self.num_patches

        for _ in range(batch_size):
            
            # target masks and their complements
            masks_p, masks_C = [], []
            
            # for every target block we want
            for _ in range(self.npred):

                ### sample a mask and also its complement
                mask, mask_C = self._sample_block_mask(p_size)

                ### store them in our list
                masks_p.append(mask)
                masks_C.append(mask_C)

                ### update how many patches in this mask, in the end 
                ### we will only keep upto min_keep_pred for every sample
                ### in the batch
                min_keep_pred = min(min_keep_pred, len(mask))

            ### Store our list of target masks
            collated_masks_pred.append(masks_p)

            ### sample our context now
            masks_e = []
            for _ in range(self.nenc):

                ### we will grab a block, but the acceptable regions is only the locations
                ### that were not selected to be targets. This is why we have the complement, 
                ### all non-selected potions are 1, selected portions are 0, and we multiply
                ### by this mask in the _sample_block_mask method
                acceptable_regions = masks_C                   
                if self.allow_overlap:                       
                    acceptable_regions = None

                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)

                ### store
                masks_e.append(mask)

                ### update min patches for same reason
                min_keep_enc = min(min_keep_enc, len(mask))

            collated_masks_enc.append(masks_e)

        ### stack it all up
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_enc = [[cm[:min_keep_enc]  for cm in cm_list] for cm_list in collated_masks_enc]
 
        target_ids = torch.stack([torch.stack(m) for m in collated_masks_pred])  # (B, npred, Nt)
        context_ids = torch.stack([torch.stack(m) for m in collated_masks_enc])   # (B, nenc,  Nc)
 
        return context_ids, target_ids
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
 
    def plot_masks(
        context_ids,
        target_ids,
        target_blocks,  
        grid_size=14,
        title="I-JEPA Masking Pattern",
        sample_idx=0
    ):
        context_ids = np.asarray(context_ids).flatten()
        target_ids  = np.unique(np.asarray(target_ids).flatten())
 
        assert len(set(context_ids.tolist()) & set(target_ids.tolist())) == 0, \
            "Context and target overlap!"
 
        # 0=unused  1=context  2=target
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        for idx in context_ids:
            r, c = divmod(int(idx), grid_size)
            grid[r, c] = 1
        for idx in target_ids:
            r, c = divmod(int(idx), grid_size)
            grid[r, c] = 2
 
        colors = ["#1a1a2e", "#4a9eff", "#ff4d6d"]
        labels = ["Unused", "Context", "Target"]
        cmap   = plt.matplotlib.colors.ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm   = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
 
        fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d0d1a")
        ax.set_facecolor("#0d0d1a")
        ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
 
        # Subtle grid lines
        for x in range(grid_size + 1):
            ax.axhline(x - 0.5, color="#ffffff22", linewidth=0.8)
            ax.axvline(x - 0.5, color="#ffffff22", linewidth=0.8)
 
        block_colors = ["#ffd166", "#06d6a0", "#f4a261", "#c77dff"]
        for b_idx, block_ids in enumerate(target_blocks):
            block_ids = np.asarray(block_ids).flatten()
            rows = [int(idx) // grid_size for idx in block_ids]
            cols = [int(idx) %  grid_size for idx in block_ids]
            r_min, r_max = min(rows), max(rows)
            c_min, c_max = min(cols), max(cols)
            color = block_colors[b_idx % len(block_colors)]
 
            rect = mpatches.FancyBboxPatch(
                (c_min - 0.5, r_min - 0.5),
                c_max - c_min + 1,
                r_max - r_min + 1,
                boxstyle="round,pad=0.05",
                linewidth=5.0,
                edgecolor=color,
                facecolor="none",
                zorder=5,
            )
            ax.add_patch(rect)
 
            # Small label in top-left corner of each box
            ax.text(c_min - 0.35, r_min - 0.35, f"T{b_idx+1}",
                    color=color, fontsize=8, fontfamily="monospace",
                    fontweight="bold", va="top", zorder=6)
 
        n_ctx = (grid == 1).sum()
        n_tgt = (grid == 2).sum()
        ax.set_title(
            f"{title} — Sample {sample_idx+1}\n"
            f"Context: {n_ctx} patches    Target: {n_tgt} patches",
            color="white", fontsize=13, pad=16, fontfamily="monospace"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
 
        legend_patches = [
            mpatches.Patch(color=colors[i], label=f"{labels[i]} ({(grid==i).sum()})")
            for i in range(3) if (grid == i).any()
        ] + [
            mpatches.Patch(edgecolor=block_colors[b % len(block_colors)],
                           facecolor="none", linewidth=2,
                           label=f"Block T{b+1}")
            for b in range(len(target_blocks))
        ]
        ax.legend(handles=legend_patches, loc="upper right", framealpha=0.35,
                  facecolor="#1a1a2e", edgecolor="#ffffff30",
                  labelcolor="white", fontsize=9, prop={"family": "monospace"})
 
        plt.tight_layout()
        plt.show()
 
    sampler = IJEPAMaskSampler(
        img_size=224, patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        nenc=1, npred=4,
    )
 
    ctx, tgt = sampler.sample(batch_size=4)

 
    ctx_flat = ctx[:, 0, :] 
    tgt_flat = tgt.flatten(1) 

    print("Intersection (should be empty):",
          set(ctx_flat[0].tolist()) & set(tgt_flat[0].tolist()))
 
    for i in range(len(ctx_flat)):
        plot_masks(
            context_ids=ctx_flat[i].numpy(),
            target_ids=tgt_flat[i].numpy(),
            target_blocks=tgt[i].numpy(),   # (npred, Nt)
            grid_size=14,
            title="I-JEPA Masking Pattern",
            sample_idx=i,
        )
 