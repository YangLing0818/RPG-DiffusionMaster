import math
from pprint import pprint
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, Resize 
import xformers
TOKENSCON = 77
TOKENS = 75



def _memory_efficient_attention_xformers(module, query, key, value):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    hidden_states = xformers.ops.memory_efficient_attention(query, key, value,attn_bias=None)
    hidden_states = module.batch_to_head_dim(hidden_states)
    return hidden_states

def main_forward_diffusers(module,hidden_states,encoder_hidden_states,divide,userpp = False,tokens=[],width = 64,height = 64,step = 0, isxl = False, inhr = None):
    context = encoder_hidden_states
    query = module.to_q(hidden_states)
    # cond, uncond =query.chunk(2)
    # query=torch.cat([cond,uncond])
    key = module.to_k(context)
    value = module.to_v(context)
    query = module.head_to_batch_dim(query)
    key = module.head_to_batch_dim(key)
    value = module.head_to_batch_dim(value)
    hidden_states=_memory_efficient_attention_xformers(module, query, key, value)
    hidden_states = hidden_states.to(query.dtype)
    # linear proj
    hidden_states = module.to_out[0](hidden_states)
    # dropout
    hidden_states = module.to_out[1](hidden_states)
    return hidden_states
    
    
    
    
    
def hook_forwards(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "Attention":
            # print(f"Attaching hook to {name}")
            module.forward = hook_forward(self, module)           


def hook_forward(self, module):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x= hidden_states
        context= encoder_hidden_states
        # print("input : ", hidden_states.size())
        # print("tokens : ", context.size())
        # print("module : ", getattr(module, self.name,None))
        height =self.h 
        width =self.w
        x_t = x.size()[1]
        scale = round(math.sqrt(height * width / x_t))
        latent_h = round(height / scale)
        latent_w = round(width / scale)
        ha, wa = x_t % latent_h, x_t % latent_w
        if ha == 0:
            latent_w = int(x_t / latent_h)
        elif wa == 0:
            latent_h = int(x_t / latent_w)

        contexts = context.clone()

        def matsepcalc(x,contexts,pn,divide):
            h_states = []
            x_t = x.size()[1]
            (latent_h,latent_w) = split_dims(x_t, height, width, self)
            

            latent_out = latent_w
            latent_in = latent_h

            tll = self.pt
            
            i = 0
            outb = None
            if self.usebase:
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    
                i = i + 1

                out = main_forward_diffusers(module, x, context, divide,userpp =True,isxl = self.isxl)

                # if self.usebase:
                outb = out.clone()
                outb = outb.reshape(outb.size()[0], latent_h, latent_w, outb.size()[2]) 

            sumout = 0

            for drow in self.split_ratio:
                v_states = []
                sumin = 0
                for dcell in drow.cols:
                    # Grabs a set of tokens depending on number of unrelated breaks.
                    context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                    # Controlnet sends extra conds at the end of context, apply it to all regions.
                    cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                    if cnet_ext > 0:
                        context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)

                    i = i + 1 + dcell.breaks
                    # if i >= contexts.size()[1]: 
                    #     indlast = True

                    out = main_forward_diffusers(module, x, context, divide,userpp = self.pn, isxl = self.isxl)
                    
                    out = out.reshape(out.size()[0], latent_h, latent_w, out.size()[2]) # convert to main shape.
                    # if indlast:
                    addout = 0
                    addin = 0
                    sumin = sumin + int(latent_in*dcell.end) - int(latent_in*dcell.start)
                    if dcell.end >= 0.999:
                        addin = sumin - latent_in
                        sumout = sumout + int(latent_out*drow.end) - int(latent_out*drow.start)
                        if drow.end >= 0.999:
                            addout = sumout - latent_out
                    out = out[:,int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                                int(latent_w*dcell.start) + addin:int(latent_w*dcell.end),:]
                    if self.usebase : 
                        # outb_t = outb[:,:,int(latent_w*drow.start):int(latent_w*drow.end),:].clone()
                        outb_t = outb[:,int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                                        int(latent_w*dcell.start) + addin:int(latent_w*dcell.end),:].clone()
                        out = out * (1 - dcell.base) + outb_t * dcell.base
            
                    v_states.append(out)

                            
                output_x = torch.cat(v_states,dim = 2) # First concat the cells to rows.

                h_states.append(output_x)
            output_x = torch.cat(h_states,dim = 1) # Second, concat rows to layer.
            output_x = output_x.reshape(x.size()[0],x.size()[1],x.size()[2]) # Restore to 3d source.  
            return output_x
        if x.size()[0] == 1 * self.batch_size:
            output_x = matsepcalc(x, contexts, self.pn, 1)
        else:
            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                nx, px = x.chunk(2)
                conn,conp = contexts.chunk(2)
            else:
                px, nx = x.chunk(2)
                conp,conn = contexts.chunk(2)
            opx = matsepcalc(px, conp, True, 2)
            onx = matsepcalc(nx, conn, False, 2)
            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                output_x = torch.cat([onx, opx])
            else:
                output_x = torch.cat([opx, onx]) 
            
            # px = x
            # conp = contexts
            # opx = matsepcalc(px, conp, True, 2)
            # output_x = opx 
        self.pn = not self.pn
        self.count = 0
        # self.count += 1

        # limit = 70 if self.isxl else 16

        # if self.count == limit:
        #     self.pn = not self.pn
        #     self.count = 0
        #     self.pfirst = False
        #     self.condi += 1
        return output_x

    return forward

def split_dims(x_t, height, width, self=None):
    """Split an attention layer dimension to height + width.
    The original estimate was latent_h = sqrt(hw_ratio*x_t),
    rounding to the nearest value. However, this proved inaccurate.
    The actual operation seems to be as follows:
    - Divide h,w by 8, rounding DOWN.
    - For every new layer (of 4), divide both by 2 and round UP (then back up).
    - Multiply h*w to yield x_t.
    There is no inverse function to this set of operations,
    so instead we mimic them without the multiplication part using the original h+w.
    It's worth noting that no known checkpoints follow a different system of layering,
    but it's theoretically possible. Please report if encountered.
    """
    scale = math.ceil(math.log2(math.sqrt(height * width / x_t)))
    latent_h = repeat_div(height, scale)
    latent_w = repeat_div(width, scale)
    if x_t > latent_h * latent_w and hasattr(self, "nei_multi"):
        latent_h, latent_w = self.nei_multi[1], self.nei_multi[0] 
        while latent_h * latent_w != x_t:
            latent_h, latent_w = latent_h // 2, latent_w // 2

    return latent_h, latent_w

def repeat_div(x,y):
    """Imitates dimension halving common in convolution operations.
    
    This is a pretty big assumption of the model,
    but then if some model doesn't work like that it will be easy to spot.
    """
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x