import colorsys  # Polygon regions.
from PIL import Image, ImageChops
from pprint import pprint
import cv2  # Polygon regions.
import gradio as gr
import numpy as np
import PIL
import torch
from modules import devices

def lange(l):
    return range(len(l))

# SBM Keywords and delimiters for region breaks, following matlab rules.
# BREAK keyword is now passed through,  
KEYROW = "ADDROW"
KEYCOL = "ADDCOL"
KEYBASE = "ADDBASE"
KEYCOMM = "ADDCOMM"
KEYBRK = "BREAK"
KEYPROMPT = "ADDP"
DELIMROW = ";"
DELIMCOL = ","
MCOLOUR = 256
NLN = "\n"
DKEYINOUT = { # Out/in, horizontal/vertical or row/col first.
("out",False): KEYROW,
("in",False): KEYCOL,
("out",True): KEYCOL,
("in",True): KEYROW,
}

ALLKEYS = [KEYCOMM,KEYROW, KEYCOL, KEYBASE, KEYPROMPT]
ALLALLKEYS = [KEYCOMM,KEYROW, KEYCOL, KEYBASE, KEYPROMPT, KEYBRK, "AND"]

fidentity = lambda x: x
ffloatd = lambda c: (lambda x: floatdef(x,c))
fcolourise = lambda: np.random.randint(0,MCOLOUR,size = 3)
fspace = lambda x: " {} ".format(x)

"""
SBM mod: Two dimensional regions (of variable size, NOT a matrix).
- Adds keywords ADDROW, ADDCOL and respective delimiters for aratios.
- A/bratios become list dicts: Inner dict of cols (varying length list) + start/end + number of breaks,
  outer layer is rows list.
  First value in each row is the row's ratio, the rest are col ratios.
  This fits prompts going left -> right, top -> down. 
- Unrelated BREAKS are counted per cell, and later extracted as multiple context indices.
- Each layer is cut up by both row + col ratios.
- Style improvements: Created classes for rows + cells and functions for some of the splitting.
- Base prompt overhaul: Added keyword ADDBASE, when present will trigger "use_base" automatically;
  base is excluded from the main prompt for dim calcs; returned to start before hook (+ base break count);
  during hook, context index skips base break count + 1. Rest is applied normally.
- To specify cols first, use "vertical" mode. eg 1st col:2 rows, 2nd col:1 row.
  In effect, this merely reverses the order of iteration for every row/col loop and whatnot.
"""

class RegionCell():
    """Cell used to split a layer to single prompts."""
    def __init__(self, st, ed, base, breaks):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.st = st # Range for the cell (cols only).
        self.ed = ed
        self.base = base # How much of the base prompt is applied (difference).
        self.breaks = breaks # How many unrelated breaks the prompt contains.
        
    def __repr__(self):
        """Debug print."""
        return "({:.2f}:{:.2f})".format(self.st,self.ed) 
        
class RegionRow():
    """Row containing cell refs and its own ratio range."""
    def __init__(self, st, ed, cols):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.st = st # Range for the row.
        self.ed = ed
        self.cols = cols # List of cells.
        
    def __repr__(self):
        """Debug print."""
        return "Outer ({:.2f}:{:.2f}), contains {}".format(self.st, self.ed, self.cols) + NLN

def floatdef(x, vdef):
    """Attempt conversion to float, use default value on error.
    
    Mainly for empty ratios, double commas.
    """
    try:
        return float(x)
    except ValueError:
        print("'{}' is not a number, converted to {}".format(x,vdef))
        return vdef

def split_l2(s, kr, kc, indsingles = False, fmap = fidentity, basestruct = None, indflip = False):
    """Split string to 2d list (ie L2) per row and col keys.
    
    The output is a list of lists, each of varying length.
    If a L2 basestruct is provided,
    will adhere to its structure using the following broadcast rules:
    - Basically matches row by row of base and new.
    - If a new row is shorter than base, the last value is repeated to fill the row.
    - If both are the same length, copied as is.
    - If new row is longer, then additional values will overflow to the next row.
      This might be unintended sometimes, but allows making all items col separated,
      then the new structure is simply adapted to the base structure.
    - If there are too many values in new, they will be ignored.
    - If there are too few values in new, the last one is repeated to fill base. 
    For mixed row + col ratios, singles flag is provided -
    will extract the first value of each row to a separate list,
    and output structure is (row L1,cell L2).
    There MUST be at least one value for row, one value for col when singles is on;
    to prevent errors, the row value is copied to col if it's alone (shouldn't affect results).
    Singles still respects base broadcast rules, and repeats its own last value.
    The fmap function is applied to each cell before insertion to L2;
    if it fails, a default value is used.
    If flipped, the keyword for columns is applied before rows.
    TODO: Needs to be a case insensitive split. Use re.split.
    """
    if indflip:
        tmp = kr
        kr = kc
        kc = tmp
    lret = []
    if basestruct is None:
        lrows = s.split(kr)
        lrows = [row.split(kc) for row in lrows]
        for r in lrows:
            cell = [fmap(x) for x in r]
            lret.append(cell)
        if indsingles:
            lsingles = [row[0] for row in lret]
            lcells = [row[1:] if len(row) > 1 else row for row in lret]
            lret = (lsingles,lcells)
    else:
        lrows = str(s).split(kr)
        r = 0
        lcells = []
        lsingles = []
        vlast = 1
        for row in lrows:
            row2 = row.split(kc)
            row2 = [fmap(x) for x in row2]
            vlast = row2[-1]
            indstop = False
            while not indstop:
                if (r >= len(basestruct) # Too many cell values, ignore.
                or (len(row2) == 0 and len(basestruct) > 0)): # Cell exhausted.
                    indstop = True
                if not indstop:
                    if indsingles: # Singles split.
                        lsingles.append(row2[0]) # Row ratio.
                        if len(row2) > 1:
                            row2 = row2[1:]
                    if len(basestruct[r]) >= len(row2): # Repeat last value.
                        indstop = True
                        broadrow = row2 + [row2[-1]] * (len(basestruct[r]) - len(row2))
                        r = r + 1
                        lcells.append(broadrow)
                    else: # Overfilled this row, cut and move to next.
                        broadrow = row2[:len(basestruct[r])]
                        row2 = row2[len(basestruct[r]):]
                        r = r + 1
                        lcells.append(broadrow)
        # If not enough new rows, repeat the last one for entire base, preserving structure.
        cur = len(lcells)
        while cur < len(basestruct):
            lcells.append([vlast] * len(basestruct[cur]))
            cur = cur + 1
        lret = lcells
        if indsingles:
            lsingles = lsingles + [lsingles[-1]] * (len(basestruct) - len(lsingles))
            lret = (lsingles,lcells)
    return lret

def is_l2(l):
    return isinstance(l[0],list) 

def l2_count(l):
    cnt = 0
    for row in l:
        cnt + cnt + len(row)
    return cnt

def list_percentify(l):
    """Convert each row in L2 to relative part of 100%. 
    
    Also works on L1, applying once globally.
    """
    lret = []
    if is_l2(l):
        for row in l:
            # row2 = [float(v) for v in row]
            row2 = [v / sum(row) for v in row]
            lret.append(row2)
    else:
        row = l[:]
        # row2 = [float(v) for v in row]
        row2 = [v / sum(row) for v in row]
        lret = row2
    return lret

def list_cumsum(l):
    """Apply cumsum to L2 per row, ie newl[n] = l[0:n].sum .
    
    Works with L1.
    Actually edits l inplace, idc.
    """
    lret = []
    if is_l2(l):
        for row in l:
            for (i,v) in enumerate(row):
                if i > 0:
                    row[i] = v + row[i - 1]
            lret.append(row)
    else:
        row = l[:]
        for (i,v) in enumerate(row):
            if i > 0:
                row[i] = v + row[i - 1]
        lret = row
    return lret

def list_rangify(l):
    """Merge every 2 elems in L2 to a range, starting from 0.  
    
    """
    lret = []
    if is_l2(l):
        for row in l:
            row2 = [0] + row
            row3 = []
            for i in range(len(row2) - 1):
                row3.append([row2[i],row2[i + 1]]) 
            lret.append(row3)
    else:
        row2 = [0] + l
        row3 = []
        for i in range(len(row2) - 1):
            row3.append([row2[i],row2[i + 1]]) 
        lret = row3
    return lret

def round_dim(x,y):
    """Return division of two numbers, rounding 0.5 up.
    
    Seems that dimensions which are exactly 0.5 are rounded up - see 680x488, second iter.
    A simple mod check should get the job done.
    If not, can always brute force the divisor with +-1 on each of h/w.
    """
    return x // y + (x % y >= y // 2)


def isfloat(t):
    try:
        float(t)
        return True
    except Exception:
        return False

def ratiosdealer(aratios2,aratios2r):
    aratios2 = list_percentify(aratios2)
    aratios2 = list_cumsum(aratios2)
    aratios2 = list_rangify(aratios2)
    aratios2r = list_percentify(aratios2r)
    aratios2r = list_cumsum(aratios2r)
    aratios2r = list_rangify(aratios2r)
    return aratios2,aratios2r

def changecs(ratios):
    ratios = ratios.replace(",","_")
    ratios = ratios.replace(";",",")
    ratios = ratios.replace("_",";")
    return ratios

def makeimgtmp(aratios,mode,usecom,usebase, flipper,ho,wo, image = None, alpha = 0.5,inprocess = False):
    if image is not None:
        wo, ho  =  image.size
    if mode == "Columns":mode = "Horizontal"
    if mode == "Rows":mode = "Vertical"

    if flipper: aratios = changecs(aratios)

    indflip = ("Ver" in mode)
    if DELIMROW not in aratios: # Commas only - interpret as 1d.
        aratios2 = split_l2(aratios, DELIMROW, DELIMCOL, fmap = ffloatd(1), indflip = False)
        aratios2r = [1]
    else:
        (aratios2r,aratios2) = split_l2(aratios, DELIMROW, DELIMCOL, 
                                        indsingles = True, fmap = ffloatd(1), indflip = indflip)

    (aratios2,aratios2r) = ratiosdealer(aratios2,aratios2r)

    size = ho * wo

    if 262144 >= size: div = 4
    elif 1048576 >= size: div = 8
    else :div = 16
    
    h, w = ho // div, wo // div

    fx = np.zeros((h,w, 3), np.uint8)
    # Base image is coloured according to region divisions, roughly.
    for (i,ocell) in enumerate(aratios2r):
        for icell in aratios2[i]:
            # SBM Creep: Colour by delta so that distinction is more reliable.
            if not indflip:
                fx[int(h*ocell[0]):int(h*ocell[1]),int(w*icell[0]):int(w*icell[1]),:] = fcolourise()
            else:
                fx[int(h*icell[0]):int(h*icell[1]),int(w*ocell[0]):int(w*ocell[1]),:] = fcolourise()
    regions = PIL.Image.fromarray(fx)
    draw = PIL.ImageDraw.Draw(regions)
    c = 0
    def coldealer(col):
        if sum(col) > 380:return "black"
        else:return "white"
    # Add region counters at the top left corner, coloured according to hue.
    for (i,ocell) in enumerate(aratios2r):
        for icell in aratios2[i]: 
            if not indflip:
                draw.text((int(w*icell[0]),int(h*ocell[0])),f"{c}",coldealer(fx[int(h*ocell[0]),int(w*icell[0])]))
            else: 
                draw.text((int(w*ocell[0]),int(h*icell[0])),f"{c}",coldealer(fx[int(h*icell[0]),int(w*ocell[0])]))
            c += 1

    regions = regions.resize((wo, ho))

    if image is not None:
        regions = ImageChops.blend(regions, image, alpha)
    
    # Create ROW+COL template from regions.
    txtkey = fspace(DKEYINOUT[("in", indflip)]) + NLN  
    lkeys = [txtkey.join([""] * len(cell)) for cell in aratios2]
    txtkey = fspace(DKEYINOUT[("out", indflip)]) + NLN
    template = txtkey.join(lkeys) 
    if usebase:
        template = fspace(KEYBASE) + NLN + template
    if usecom:
        template = fspace(KEYCOMM) + NLN + template

    if inprocess:
        changer = template.split(NLN)
        changer = [l.strip() for l in changer]
        return changer
    
    return regions, gr.update(value = template)

################################################################
##### matrix
fcountbrk = lambda x: x.count(KEYBRK)
fint = lambda x: int(x)

def matrixdealer(self, p, aratios, bratios, mode):
    print(aratios, bratios, mode)
    if "Ran" in mode:
        randdealer(self,p,aratios,bratios)
        return 
    # The addrow/addcol syntax is better, cannot detect regular breaks without it.
    # In any case, the preferred method will anchor the L2 structure. 
    # No prompt formatting is performed. Used only for region calculations
    prompt = p.prompt
    if self.debug: print("in matrixdealer",prompt)
    if KEYCOMM in prompt: prompt = prompt.split(KEYCOMM,1)[1]
    if KEYBASE in prompt: prompt = prompt.split(KEYBASE,1)[1]

    indflip = ("Ver" in mode)
    if (KEYCOL in prompt.upper() or KEYROW in prompt.upper()):
        breaks = prompt.count(KEYROW) + prompt.count(KEYCOL) + int(self.usebase)
        # Prompt anchors, count breaks between special keywords.
        lbreaks = split_l2(prompt, KEYROW, KEYCOL, fmap = fcountbrk, indflip = indflip)
        if (DELIMROW not in aratios
        and (KEYROW in prompt.upper()) != (KEYCOL in prompt.upper())):
            # By popular demand, 1d integrated into 2d.
            # This works by either adding a single row value (inner),
            # or setting flip to the reverse (outer).
            # Only applies when using just ADDROW / ADDCOL keys, and commas in ratio.
            indflip2 = False
            if (KEYROW in prompt.upper()) == indflip:
                aratios = "1" + DELIMCOL + aratios
            else:
                indflip2 = True
            (aratios2r,aratios2) = split_l2(aratios, DELIMROW, DELIMCOL, indsingles = True,
                                fmap = ffloatd(1), basestruct = lbreaks,
                                indflip = indflip2)
        else: # Standard ratios, split to rows and cols.
            (aratios2r,aratios2) = split_l2(aratios, DELIMROW, DELIMCOL, indsingles = True,
                                            fmap = ffloatd(1), basestruct = lbreaks, indflip = indflip)
        # More like "bweights", applied per cell only.
        bratios2 = split_l2(bratios, DELIMROW, DELIMCOL, fmap = ffloatd(0), basestruct = lbreaks, indflip = indflip)
    else:
        breaks = prompt.count(KEYBRK) + int(self.usebase)
        (aratios2r,aratios2) = split_l2(aratios, DELIMROW, DELIMCOL, indsingles = True, fmap = ffloatd(1), indflip = indflip)
        # Cannot determine which breaks matter.
        lbreaks = split_l2("0", KEYROW, KEYCOL, fmap = fint, basestruct = aratios2, indflip = indflip)
        bratios2 = split_l2(bratios, DELIMROW, DELIMCOL, fmap = ffloatd(0), basestruct = lbreaks, indflip = indflip)
        # If insufficient breaks, try to broadcast prompt - a bit dumb.
        breaks = fcountbrk(prompt)
        lastprompt = prompt.rsplit(KEYBRK)[-1]
        if l2_count(aratios2) > breaks: 
            prompt = prompt + (fspace(KEYBRK) + lastprompt) * (l2_count(aratios2) - breaks) 
    (aratios,aratiosr) = ratiosdealer(aratios2,aratios2r)
    bratios = bratios2 
    
    # Merge various L2s to cells and rows.
    drows = []
    for r,_ in enumerate(lbreaks):
        dcells = []
        for c,_ in enumerate(lbreaks[r]):
            d = RegionCell(aratios[r][c][0], aratios[r][c][1], bratios[r][c], lbreaks[r][c])
            dcells.append(d)
        drow = RegionRow(aratiosr[r][0], aratiosr[r][1], dcells)
        drows.append(drow)

    self.aratios = drows
    self.bratios = bratios

################################################################
##### inpaint

"""
SBM mod: Mask polygon region.
- Basically a version of inpainting, where polygon outlines are drawn and added to a coloured image.
- Colours from the image are picked apart for masks corresponding to regions.
- In new mask mode, masks are stored instead of aratios, and applied to each region forward.
- Mask can be uploaded (alpha, no save), and standard colours are detected from it.
- Uncoloured regions default to the first colour detected;
  however, if base mode is used, instead base will be applied to the remainder at 100% strength.
  I think this makes it far more useful. At 0 strength, it will apply ONLY to said regions.
- V2: Corrects and detects colours from upload.
- Mask mode presets save mask to a file, which is loaded with the preset.
- Added -1 colour to clear sections, an eraser.
"""

POLYFACTOR = 1.5 # Small lines are detected as shapes.
COLREG = None # Computed colour regions cache. Array. Extended whenever a new colour is requested.
REGUSE = dict() # Used regions. Reset on new canvas / upload (preset). 
IDIM = 512
CBLACK = 255
MAXCOLREG = 360 - 1 # Hsv goes by degrees.
VARIANT = 0 # Ensures that the sketch canvas is actually refreshed.
# Permitted hsv error range for mask upload (due to compression).
# Mind, wrong hue might throw off the mask entirely and is not corrected.
# HSV_RANGE = (125,130)
# HSV_VAL = 128
HSV_RANGE = (0.49,0.51)
HSV_VAL = 0.5
CCHANNELS = 3
COLWHITE = (255,255,255)
# Optional replacement mode of nonstandard colours from the mask during upload with white.
# Pros: Clear and obvious display of regions.
# Cons: Cannot use the image as a background for tracing (eg openpose or depthmap).
# Compromise: Do not replace, but show the used regions.
INDCOLREPL = False

def get_colours(img):
    """List colours used in image (as nxc array).
    
    """
    return np.unique(img.reshape(-1, img.shape[-1]), axis=0)

def generate_unique_colours(n):
    """Generate n visually distinct colors as a list of RGB tuples.
    
    Uses the hue of hsv, with balanced saturation & value.
    """
    hsv_colors = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    rgb_colors = [tuple(int(i * CBLACK) for i in colorsys.hsv_to_rgb(*hsv)) for hsv in hsv_colors]
    return rgb_colors

def deterministic_colours(n, lcol = None):
    """Generate n visually distinct & consistent colours as a list of RGB tuples.
    
    Uses the hue of hsv, with balanced saturation & value.
    Goes around the cyclical 0-256 and picks each /2 value for every round.
    Continuation rules: If pcyv != ccyv in next round, then we don't care.
    If pcyv == ccyv, we want to get the cval + delta of last elem.
    If lcol > n, will return it as is.
    """
    if n <= 0:
        return None
    pcyc = -1
    cval = 0
    if lcol is None:
        st = 0
    elif n <= len(lcol):
        # return lcol[:n] # Truncating the list is accurate, but pointless.
        return lcol
    else:
        st = len(lcol)
        if st > 0:
            pcyc = np.ceil(np.log2(st))
            # This is erroneous on st=2^n, but we don't care.
            dlt = 1 / (2 ** pcyc)
            cval = dlt + 2 * dlt * (st % (2 ** (pcyc - 1)) - 1)

    lhsv = []
    for i in range(st,n):
        ccyc = np.ceil(np.log2(i + 1))
        if ccyc == 0: # First col = 0.
            cval = 0
            pcyc = ccyc
        elif pcyc != ccyc: # New cycle, start from the half point between 0 and first point.
            dlt = 1 / (2 ** ccyc)
            cval = dlt
            pcyc = ccyc
        else:
            cval = cval + 2 * dlt # Jumps over existing vals.
        lhsv.append(cval)
    lhsv = [(v, 0.5, 0.5) for v in lhsv] # Hsv conversion only works 0:1.
    lrgb = [colorsys.hsv_to_rgb(*hsv) for hsv in lhsv]
    lrgb = (np.array(lrgb) * (CBLACK + 1)).astype(np.uint8) # Convert to colour uints.
    lrgb = lrgb.reshape(-1, CCHANNELS)
    if lcol is not None:
        lrgb = np.concatenate([lcol, lrgb])
    return lrgb

def index_rows(mat):
    """In 2D matrix, add column containing row number.
    
    Pandas stuff, can't find a clever way to find first row in np.
    """
    return np.concatenate([np.arange(len(mat)).reshape(-1,1),mat],axis = 1)

def detect_image_colours(img, inddict = False):
    """Detect relevant hsv colours in image and clean up the standard mask.
    
    Basically, converts colours to hsv, checks which ones are within range,
    converts them to the exact sv value we need, deletes irrelevant colours,
    and creates a list of used colours via a form of np first row lookup.
    Problem: Rgb->hsb and back is not lossless in np / cv. Getting 128->127.
    Looks like the only option is to use colorsys which is contiguous.
    To maximise efficiency, I've applied it to the unique colours instead of entire image,
    and then each colour is mapped via np masking (propagation),
    by adding a third fake dim for each of colours, flattened image.
    It might be possible to use cv2 one way for the filter, but I think that's risky,
    and likely doesn't save much processing (heaviest op is get_colours for large image).
    Creep: Apply erosion so thin regions are ignored. This would need be applied on processing as well.
    """
    global REGUSE
    global COLREG
    global VARIANT
    if img is None: # Do nothing if no image passed. 
        return None, None
    VARIANT = 0 # Upload doesn't need variance, it refreshes automatically.
    (h,w,c) = img.shape
    # Get unique colours, create rgb-hsv mapping and filtering.
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # skimg = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    lrgb = get_colours(img)
    lhsv = np.apply_along_axis(lambda x: colorsys.rgb_to_hsv(*x), axis=-1, arr = lrgb / CBLACK)
    msk = ((lhsv[:,1] >= HSV_RANGE[0]) & (lhsv[:,1] <= HSV_RANGE[1]) &
           (lhsv[:,2] >= HSV_RANGE[0]) & (lhsv[:,2] <= HSV_RANGE[1]))
    lfltrgb = lrgb[msk]
    lflthsv = lhsv[msk]
    lflthsv[:,1:] = HSV_VAL
    if len(lfltrgb) > 0:
        lfltfix = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), axis=-1, arr=lflthsv)
        lfltfix = (lfltfix * (CBLACK + 1)).astype(np.uint8)
    else: # No relevant colours.
        lfltfix = lfltrgb
    # Mask update each colour in the image.
    # I tried to use isin, but it seems to detect any permutation.
    # It's better to roll colour channel to the front, add extra fake dims,
    # then use direct comparison, relying on np broadcasting.
    # Shape: colour x search x img
    cnt = len(lfltrgb)
    img2 = img.reshape(-1,c,1)
    img2 = np.moveaxis(img2,0,-1)
    lfltrgb2 = np.moveaxis(lfltrgb,-1,0)
    lfltrgb2 = lfltrgb2.reshape(c,-1,1)
    msk2 = (img2 == lfltrgb2).all(axis = 0).reshape(cnt,h,w)
    for i,_ in enumerate(lfltrgb):
        img[msk2[i]] = lfltfix[i]
    # Empty all nonfiltered regions.
    msk3 = ~(msk2.any(axis = 0))
    if INDCOLREPL: # Don't remove nonstandard.
        img[msk3] = COLWHITE
    # Gen all colours, match with the fixed filtered list.
    # I can think of no mathematical function to inverse the colour gen function.
    # Also, imperfect hash, so ~60 colours go over the edge. Should have 100% matches at x2. 
    COLREG = deterministic_colours(2 * MAXCOLREG, COLREG)
    cow = index_rows(COLREG)
    regrows = [cow[(COLREG == f).all(axis = 1)] for f in lfltfix]
    # MAX_KEY_VALUE provides a threshold value. Only those colors are added to REGUSE, for which the key values 
    # (i.e., the indices of the colors in the 'regrows' array) are less than this threshold.
    # Colors with indices greater than MAX_KEY_VALUE are considered "similar colors" and are not treated as separate masks. 
    unique_keys = set(reg[0,0] for reg in regrows if len(reg) > 0)
    # The purpose of this is to reduce the number of colors being processed, particularly for colors that are 
    # close to each other, which may be slightly different due to noise in the image or minor differences in color encoding.
    # By setting an appropriate MAX_KEY_VALUE, these minor color differences can be effectively filtered out, 
    # thereby reducing the number of colors being processed and making color processing more accurate and efficient.
    MAX_KEY_VALUE = len(unique_keys) + 20
    REGUSE = {reg[0,0]: reg[0,1:].tolist() for reg in regrows if len(reg) > 0 and reg[0,0] <= MAX_KEY_VALUE}
    # REGUSE.discard(COLWHITE)
    
    # Must set to dict due to gradio preprocess assertion, in preset load.
    # CONT: This doesn't work. Postprocess expects image. Maybe use dict for preset, not upload.
    if inddict:
        img = {"image":img, "mask":None}
    
    return img, None # Clears the upload area. A bit cleaner.

def save_mask(img, flpath):
    """Save mask to file.
    
    These will be loaded as part of a preset.
    Cv's colour scheme is an annoyance, but avoiding yet another import. 
    """
    # Cv's colour scheme is annoying.
    try:
        img = img["image"]
    except Exception:
        pass
    if VARIANT != 0: # Always save without variance.
        img = img[:-VARIANT,:-VARIANT,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(flpath, img)

def load_mask(flpath):
    """Load mask from file.
    
    Does not edit mask automatically (detect colours).
    """
    try:
        img = cv2.imread(flpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception: # Could not load mask.
        img = None
    return img

def detect_polygons(img,num):
    """Convert stroke + region to standard coloured mask.
    
    Negative colours will clear the mask instead, and not ++.
    """
    global COLREG
    global VARIANT
    global REGUSE

    # I dunno why, but mask has a 4th colour channel, which contains nothing. Alpha?
    if VARIANT != 0:
        out = img["image"][:-VARIANT,:-VARIANT,:CCHANNELS]
        img = img["mask"][:-VARIANT,:-VARIANT,:CCHANNELS]
    else:
        out = img["image"][:,:,:CCHANNELS]
        img = img["mask"][:,:,:CCHANNELS]

    # Convert the binary image to grayscale
    if img is None:
        img = np.zeros([IDIM,IDIM,CCHANNELS],dtype = np.uint8) + CBLACK # Stupid cv.
    if out is None:
        out = np.zeros_like(img) + CBLACK # Stupid cv.
    bimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find contours in the image
    # Must reverse colours, otherwise draws an outer box (0->255). Dunno why gradio uses 255 for white anyway. 
    contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #img2 = np.zeros_like(img) + 255 # Fresh image.
    img2 = out # Update current image.

    if num < 0:
        color = COLWHITE
    else:
        COLREG = deterministic_colours(int(num) + 1, COLREG)
        color = COLREG[int(num),:]
        REGUSE[num] = color.tolist()
    # Loop through each contour and detect polygons
    for cnt in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(cnt, 0.0001 * cv2.arcLength(cnt, True), True)

        # If the polygon has 3 or more sides and is fully enclosed, fill it with a random color
        # if len(approx) >= 3: # BAD test.
        if cv2.contourArea(cnt) > cv2.arcLength(cnt, True) * POLYFACTOR: # Better, still messes up on large brush.
            #SBM BUGGY, prevents contours from . cv2.pointPolygonTest(approx, (approx[0][0][0], approx[0][0][1]), False) >= 0:
            
            # Draw the polygon on the image with a new random color
            color = [int(v) for v in color] # Opencv is dumb / C based and can't handle an int64 array.
            #cv2.drawContours(img2, [approx], 0, color = color) # Only outer sketch.
            cv2.fillPoly(img2,[approx],color = color)

    # Convert the grayscale image back to RGB
    #img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB) # Converting to grayscale is dumb.

    skimg = create_canvas(img2.shape[0], img2.shape[1], indwipe = False)
    if VARIANT != 0:
        skimg[:-VARIANT,:-VARIANT,:] = img2
    else:
        skimg[:,:,:] = img2
    print("Region sketch size", skimg.shape)    
    return skimg, num + 1 if (num >= 0 and num + 1 <= CBLACK) else num

def detect_mask(img, num, mult = CBLACK):
    """Extract specific colour and return mask.
    
    Multiplier for correct display.
    Also tags colour in case someone uses the upload interface. 
    """
    global REGUSE
    try:
        img = img["image"]
    except Exception:
        pass
    if img is None:
        return None
    indnot = False
    if num < 0: # Detect unmasked region.
        if INDCOLREPL: # In replacement mode, all colours are either region or white.
            color = np.array(COLWHITE).reshape([1,1,CCHANNELS])
        else: # In nonrepl mode, mask all the regions and invert.
            color = np.array(list(REGUSE.values())) # nx3
            color = np.moveaxis(color,-1,0) # 3xn
            color = color.reshape(1,1,*color.shape) # 1x1x3xn
            img = img.reshape(*img.shape,1) # Same.
            indnot = True
    else:
        color = deterministic_colours(int(num) + 1)[-1]
        color = color.reshape([1,1,CCHANNELS])
    if indnot: # Negation of a list of regions.
        mask = (~(img == color)).all(-1).all(-1)
        mask = mask * mult
    else:
        mask = ((img == color).all(-1)) * mult
    if mask.sum() > 0 and num >= 0:
        REGUSE[num] = color.reshape(-1).tolist()
    return mask

def draw_region(img, num):
    """Simply runs polygon detection, followed by mask on result.
    
    Saves extra inconvenient button. Since num is auto incremented, we take the old val.
    """
    img, num2 = detect_polygons(img, num)
    mask = detect_mask(img, num)
    # Gradio is stupid, I have to force feed it a dict so preprocess doesn't break.
    # Disabled here, can only be fixed reliably in preprocess.
    # dimg = {"image":img, "mask": None}
    dimg = img
    return dimg, num2, mask

def draw_image(img, inddict = False):
    """Runs colour detection followed by mask on -1 to show which colours are regions.
    
    """
    img, clearer = detect_image_colours(img,inddict)
    mask = detect_mask(img, -1)
    dimg = img
    return dimg, clearer, mask

def create_canvas(h, w, indwipe = True):
    """New region sketch area.
    
    Small variant value is added (and ignored later) due to gradio refresh bug.
    Meant to be used only to start over or when the image dims change.
    """
    global VARIANT
    global REGUSE
    VARIANT = 1 - VARIANT
    if indwipe:
        REGUSE = dict()
    vret =  np.zeros(shape = (h + VARIANT, w + VARIANT, CCHANNELS), dtype = np.uint8) + CBLACK
    return vret

# SBM In mask mode, grabs each mask from coloured mask image.
# If there's no base, remainder goes to first mask.
# If there's a base, it will receive its own remainder mask, applied at 100%.
def inpaintmaskdealer(self, p, bratios, usebase, polymask):
    prompt = p.prompt
    if self.debug: print("in inpaintmaskdealer",prompt)
    if KEYCOMM in prompt: prompt = prompt.split(KEYCOMM,1)[1]
    if KEYBASE in prompt: prompt = prompt.split(KEYBASE,1)[1]
    # Prep masks.
    self.regmasks = []
    tm = None
    # Sort colour dict by key, return value for masking.
    #for _,c in sorted(REGUSE.items(), key = lambda x: x[0]):
    for c in sorted(REGUSE.keys()):
        m = detect_mask(polymask, c, 1)
        if VARIANT != 0:
            m = m[:-VARIANT,:-VARIANT]
        if m.any():
            if tm is None:
                tm = np.zeros_like(m) # First mask is ignored deliberately.
                if self.usebase: # In base mode, base gets the outer regions.
                    tm = tm + m
            else:
                tm = tm + m
            m = m.reshape([1, *m.shape]).astype(np.float16)
            t = torch.from_numpy(m).to(devices.device) 
            self.regmasks.append(t)
    # First mask applies to all unmasked regions.
    m = 1 - tm
    m = m.reshape([1, *m.shape]).astype(np.float16)
    t = torch.from_numpy(m).to(devices.device)
    if self.usebase:
        self.regbase = t
    else:
        self.regbase = None
        self.regmasks[0] = t
    
    # Simulated region anchroing for base weights.
    breaks = prompt.count(KEYBRK)
    self.bratios = split_l2(bratios, DELIMROW, DELIMCOL, fmap = ffloatd(0),
                            basestruct = [[0] * (breaks + 1)], indflip = False)

def randdealer(self,p,aratios,bratios):
    # h*w の大きさのテンソルを作成
    tensor = torch.zeros((p.height//8, p.width//8)).to("cuda")
    x,y = int(aratios.split(",")[0]),int(aratios.split(",")[1])



    # 領域ごとのサイズを計算
    dh, dw = p.height//8 // x, p.width//8 // y
    lbreaks = p.prompt.count(KEYBRK) + 1

    bratios = bratios.split(",") if self.usebase else [0]
    bratios = [float(b) for b in bratios]
    while len(bratios) <= lbreaks:
        bratios.append(bratios[0])

    # 領域ごとに0から3までのランダムな値を設定
    for i in range(x):
        for j in range(y):
            random_value = torch.randint(0, lbreaks, (1,))
            tensor[i*dh:(i+1)*dh, j*dw:(j+1)*dw] = random_value
    tensors = []

    ranbase = torch.ones_like(tensor)

    for i in range(lbreaks):
        add = torch.where(tensor==i, 1*(1-bratios[i]),0)
        tensors.append(add)
        ranbase = ranbase - add

    drows = []
    dcells = []
    for c in range(lbreaks):
        d = RegionCell(0,0 , 0, 0)
        dcells.append(d)
    drow = RegionRow(0, 1, dcells)
    drows.append(drow)

    self.aratios = drows
    self.ransors = tensors
    self.ranbase = ranbase
