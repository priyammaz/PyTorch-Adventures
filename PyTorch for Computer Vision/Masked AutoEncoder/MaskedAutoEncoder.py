import torch
import torch.nn as nn
from VisionTransformer import PatchEmbed, TransformerBlock

"""
Work in Progress! Reference code here:  https://github.com/facebookresearch/mae/blob/main/models_mae.py
"""
class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, encoder_embed_dim=768,
                 encoder_depth=12, encoder_num_heads=12, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16):

        super(MaskedAutoencoder, self).__init__()

        self.patch_size = patch_size

        ### Define Encoder Parts ###
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=encoder_embed_dim)
        self.enc_cls_token = nn.Parameter(torch.zeros(1,1,encoder_embed_dim))
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim=encoder_embed_dim,
                                 num_heads=encoder_num_heads,
                                 efficient=True)
                for _ in range(encoder_depth)
            ]
        )
        self.enc_norm = nn.LayerNorm(encoder_embed_dim)


        ### Map Encoder Embed Dim to Decoder Embed Dim
        self.enc2dec_mapping = nn.Linear(encoder_embed_dim, decoder_embed_dim)

        ### Define Decoder Parts ###
        self.mask_token = nn.Parameter(torch.zeros(1,1,decoder_embed_dim))

        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim=decoder_embed_dim,
                                 num_heads=decoder_num_heads,
                                 efficient=True)
                for _ in range(decoder_depth)
            ]
        )

        self.dec_norm = nn.LayerNorm(decoder_embed_dim)

        ### Output Decoder to Prediction Size
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)

    def patchify(self, imgs):
        h = w = imgs.shape[2] // self.patch_size
        x = imgs.reshape(imgs.shape[0], 3, h, self.patch_size, w, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(imgs.shape[0], h*w, self.patch_size**2*3)
        return x


    def random_masking(self, x, mask_ratio):
        batch, seq_len, embed_dim = x.shape
        len_keep = int(seq_len * (1 - mask_ratio))
        noise = torch.rand(batch, seq_len)

        # Sort Noise per sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]


        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,embed_dim))
        mask = torch.ones([batch, seq_len])
        mask[:, :len_keep] = 0 # 0 is keep, 1 is remove

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_enc(self, x, mask_ratio):
        x = self.patch_embed(x)

        ### Add once without CLS token ###
        x = x + self.enc_pos_embed[:, 1, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.enc_cls_token + self.enc_pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)

        x = self.enc_norm(x)
        return x, mask, ids_restore

    def forward_dec(self, x, ids_restore):
        x = self.enc2dec_mapping(x)
        mask_token = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_token], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,x.shape[-1]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.dec_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.dec_norm(x)

        x = self.decoder_pred(x)

        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, preds, mask):
        target = self.patchify(imgs)
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss


    def forward(self, x, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_enc(x, mask_ratio)
        pred = self.forward_dec(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask




mae = MaskedAutoencoder()
rand = torch.rand(2, 3, 224, 224)
loss, pred, mask = mae(rand)



