# Copyright 2024 ByteDance and/or its affiliates.
#
# Copyright (2024) X-Dyna Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.

import torch
from torch import nn
from einops import rearrange
class ModelWrapper(nn.Module):
    def __init__(self, unet, controlnet, controlnet_xbody=None, face_image_proj_model=None):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.controlnet_xbody = controlnet_xbody
        self.face_image_proj_model = face_image_proj_model

    def forward(self,
            timesteps,
            noisy_latents, 
            unet_encoder_hidden_states, 
            encoder_hidden_states,
            controlnet_condition,
            controlnet_xbody_condition=None,
            face_emb=None,
            conditioning_scale=1.0,
            return_dict=False,
            cross_id=False,
        ):
        if (face_emb is not None) and (self.face_image_proj_model is not None): # use face ip
            face_tokens = self.face_image_proj_model(face_emb)  
            unet_encoder_hidden_states = torch.cat([unet_encoder_hidden_states, face_tokens], dim=1)
        b, c, f, h, w = noisy_latents.shape
        if cross_id:
            f = f - 1 # 16
            controlnet_latent_input = rearrange(noisy_latents[:,:,1:,:,:], "b c f h w -> (b f) c h w")  
        else:
            controlnet_latent_input = rearrange(noisy_latents, "b c f h w -> (b f) c h w")  
        down_block_res_samples, mid_block_res_sample = self.controlnet(
                controlnet_latent_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_condition,
                conditioning_scale=conditioning_scale,
                return_dict=return_dict,
        )
        if (controlnet_xbody_condition is not None) and (self.controlnet_xbody is not None):
            down_block_res_samples_xbody, mid_block_res_sample_xbody = self.controlnet_xbody(
                    controlnet_latent_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_xbody_condition,
                    conditioning_scale=conditioning_scale,
                    return_dict=return_dict,
            )            
        # reshape controlnet output to match the unet3d inputs
        _down_block_res_samples = []
        if (controlnet_xbody_condition is not None) and (self.controlnet_xbody is not None):
            for sample, sample_xbody in zip(down_block_res_samples, down_block_res_samples_xbody):
                sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=f)
                sample_xbody = rearrange(sample_xbody, '(b f) c h w -> b c f h w', b=b, f=f)
                B, C, Frame, H, W = sample.shape
                if cross_id:
                    sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), sample],dim=2) # b c 17 h w
                    sample_xbody = torch.cat([torch.zeros(B,C,1,H,W).to(sample_xbody.device,sample_xbody.dtype), sample_xbody],dim=2) # b c 17 h w
                    sample_sum = sample + sample_xbody
                _down_block_res_samples.append(sample_sum)
            down_block_res_samples = _down_block_res_samples
            mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=f)
            mid_block_res_sample_xbody = rearrange(mid_block_res_sample_xbody, '(b f) c h w -> b c f h w', b=b, f=f)
            B, C, Frame, H, W = mid_block_res_sample.shape
            if cross_id:
                mid_block_res_sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), mid_block_res_sample],dim=2) # b c 17 h w
                mid_block_res_sample_xbody = torch.cat([torch.zeros(B,C,1,H,W).to(sample_xbody.device,sample_xbody.dtype), mid_block_res_sample_xbody],dim=2) # b c 17 h w
                mid_block_res_sample += mid_block_res_sample_xbody
        else:
            for sample in down_block_res_samples:
                sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=f)
                B, C, Frame, H, W = sample.shape
                if cross_id:
                    sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), sample],dim=2) # b c 17 h w
                _down_block_res_samples.append(sample)
            down_block_res_samples = _down_block_res_samples
            mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=f)
            B, C, Frame, H, W = mid_block_res_sample.shape
            if cross_id:
                mid_block_res_sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), mid_block_res_sample],dim=2) # b c 17 h w
    
        model_pred = self.unet(noisy_latents, timesteps, unet_encoder_hidden_states, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample).sample
        
        return model_pred

# Copied from CameraCtrl
class PoseAdaptor(nn.Module):
    def __init__(self, unet, pose_encoder):
        super().__init__()
        self.unet = unet
        self.pose_encoder = pose_encoder

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, pose_embedding):
        assert pose_embedding.ndim == 5
        bs = pose_embedding.shape[0]            # b c f h w
        pose_embedding_features = self.pose_encoder(pose_embedding)      # bf c h w
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                   for x in pose_embedding_features]
        noise_pred = self.unet(noisy_latents,
                               timesteps,
                               encoder_hidden_states,
                               pose_embedding_features=pose_embedding_features).sample
        return noise_pred


class ModelWrapper_Camera(nn.Module):
    def __init__(self, unet, controlnet, pose_encoder, controlnet_xbody=None):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.pose_encoder = pose_encoder
        self.controlnet_xbody = controlnet_xbody


    def forward(self,
            timesteps,
            noisy_latents, 
            unet_encoder_hidden_states, 
            encoder_hidden_states,
            controlnet_condition,
            pose_embedding,
            controlnet_xbody_condition=None,
            conditioning_scale=1.0,
            return_dict=False,
            cross_id=False,
        ):

        assert pose_embedding.ndim == 5
        bs = pose_embedding.shape[0]            # b c f h w
        pose_embedding_features = self.pose_encoder(pose_embedding)      # bf c h w
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                   for x in pose_embedding_features]

        b, c, f, h, w = noisy_latents.shape
        if cross_id:
            f = f - 1 # 16
            controlnet_latent_input = rearrange(noisy_latents[:,:,1:,:,:], "b c f h w -> (b f) c h w")  
        else:
            controlnet_latent_input = rearrange(noisy_latents, "b c f h w -> (b f) c h w")  
        down_block_res_samples, mid_block_res_sample = self.controlnet(
                controlnet_latent_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_condition,
                conditioning_scale=conditioning_scale,
                return_dict=return_dict,
        )
        if (controlnet_xbody_condition is not None) and (self.controlnet_xbody is not None):
            down_block_res_samples_xbody, mid_block_res_sample_xbody = self.controlnet_xbody(
                    controlnet_latent_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_xbody_condition,
                    conditioning_scale=conditioning_scale,
                    return_dict=return_dict,
            )            
        # reshape controlnet output to match the unet3d inputs
        _down_block_res_samples = []
        if (controlnet_xbody_condition is not None) and (self.controlnet_xbody is not None):
            for sample, sample_xbody in zip(down_block_res_samples, down_block_res_samples_xbody):
                sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=f)
                sample_xbody = rearrange(sample_xbody, '(b f) c h w -> b c f h w', b=b, f=f)
                B, C, Frame, H, W = sample.shape
                if cross_id:
                    sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), sample],dim=2) # b c 17 h w
                    sample_xbody = torch.cat([torch.zeros(B,C,1,H,W).to(sample_xbody.device,sample_xbody.dtype), sample_xbody],dim=2) # b c 17 h w
                    sample_sum = sample + sample_xbody
                _down_block_res_samples.append(sample_sum)
            down_block_res_samples = _down_block_res_samples
            mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=f)
            mid_block_res_sample_xbody = rearrange(mid_block_res_sample_xbody, '(b f) c h w -> b c f h w', b=b, f=f)
            B, C, Frame, H, W = mid_block_res_sample.shape
            if cross_id:
                mid_block_res_sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), mid_block_res_sample],dim=2) # b c 17 h w
                mid_block_res_sample_xbody = torch.cat([torch.zeros(B,C,1,H,W).to(sample_xbody.device,sample_xbody.dtype), mid_block_res_sample_xbody],dim=2) # b c 17 h w
                mid_block_res_sample += mid_block_res_sample_xbody
        else:
            for sample in down_block_res_samples:
                sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=f)
                B, C, Frame, H, W = sample.shape
                if cross_id:
                    sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), sample],dim=2) # b c 17 h w
                _down_block_res_samples.append(sample)
            down_block_res_samples = _down_block_res_samples
            mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=f)
            B, C, Frame, H, W = mid_block_res_sample.shape
            if cross_id:
                mid_block_res_sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), mid_block_res_sample],dim=2) # b c 17 h w

        model_pred = self.unet(noisy_latents, timesteps, unet_encoder_hidden_states, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample, pose_embedding_features=pose_embedding_features).sample
        
        return model_pred
