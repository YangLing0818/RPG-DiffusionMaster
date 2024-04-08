import colorsys  # Polygon regions.
from PIL import Image, ImageChops
from pprint import pprint
import cv2  # Polygon regions.
import numpy as np
import PIL
import torch

SPLROW = ";"
SPLCOL = ","
KEYROW = "ADDROW"
KEYCOL = "ADDCOL"
KEYBASE = "ADDBASE"
KEYCOMM = "ADDCOMM"
KEYBRK = "BREAK"
NLN = "\n"
DKEYINOUT = { # Out/in, horizontal/vertical or row/col first.
("out",False): KEYROW,
("in",False): KEYCOL,
("out",True): KEYCOL,
("in",True): KEYROW,
}
fidentity = lambda x: x
ffloatd = lambda c: (lambda x: floatdef(x,c))
fspace = lambda x: " {} ".format(x)
fcountbrk = lambda x: x.count(KEYBRK)
fint = lambda x: int(x)

def floatdef(x, vdef):
    """Attempt conversion to float, use default value on error.    
    Mainly for empty ratios, double commas.
    """
    try:
        return float(x)
    except ValueError:
        print("'{}' is not a number, converted to {}".format(x,vdef))
        return vdef

class Region():
    """Specific Region used to split a layer to single prompts."""
    def __init__(self, st, ed, base, breaks):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.start = st # Range for the cell (cols only).
        self.end = ed
        self.base = base # How much of the base prompt is applied (difference).
        self.breaks = breaks # How many unrelated breaks the prompt contains.

class Row():
    """Row containing cell refs and its own ratio range."""
    def __init__(self, st, ed, cols):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.start = st # Range for the row.
        self.end = ed
        self.cols = cols # List of cells.
        
def is_l2(l):
    return isinstance(l[0],list) 

def l2_count(l):
    cnt = 0
    for row in l:
        cnt + cnt + len(row)
    return cnt

def list_percentify(l):
    """
    Convert each row in L2 to relative part of 100%. 
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
    """
    Apply cumsum to L2 per row, ie newl[n] = l[0:n].sum .
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
    """
    Merge every 2 elems in L2 to a range, starting from 0.  
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

def ratiosdealer(split_ratio2,split_ratio2r):
    split_ratio2 = list_percentify(split_ratio2)
    split_ratio2 = list_cumsum(split_ratio2)
    split_ratio2 = list_rangify(split_ratio2)
    split_ratio2r = list_percentify(split_ratio2r)
    split_ratio2r = list_cumsum(split_ratio2r)
    split_ratio2r = list_rangify(split_ratio2r)
    return split_ratio2,split_ratio2r

def round_dim(x,y):
    """Return division of two numbers, rounding 0.5 up.    
    Seems that dimensions which are exactly 0.5 are rounded up - see 680x488, second iter.
    A simple mod check should get the job done.
    If not, can always brute force the divisor with +-1 on each of h/w.
    """
    return x // y + (x % y >= y // 2)       

def keyconverter(self,split_ratio,usebase):
    '''convert BREAKS to ADDCOMM/ADDBASE/ADDCOL/ADDROW'''
    if SPLROW not in split_ratio: # Commas only - interpret as 1d.
        split_ratio2 = split_l2(split_ratio, SPLROW, SPLCOL, map_function = ffloatd(1))
        split_ratio2r = [1]
    else:
        (split_ratio2r,split_ratio2) = split_l2(split_ratio, SPLROW, SPLCOL, 
                                        indsingles = True, map_function = ffloatd(1))
    (split_ratio2,split_ratio2r) = ratiosdealer(split_ratio2,split_ratio2r)
    #print(keychanger,p.prompt)
    txtkey = fspace(DKEYINOUT[("in", False)]) + NLN  
    lkeys = [txtkey.join([""] * len(cell)) for cell in split_ratio2]
    txtkey = fspace(DKEYINOUT[("out", False)]) + NLN
    template = txtkey.join(lkeys) 
    if usebase:
        template = fspace(KEYBASE) + NLN + template
    changer = template.split(NLN)
    changer = [l.strip() for l in changer]
    keychanger=changer[:-1]
    for change in keychanger:
        if change == KEYBASE and KEYBASE in self.prompt: continue
        self.prompt= self.prompt.replace(KEYBRK,change,1)                        

def split_l2(s, key_row, key_col, indsingles = False, map_function = fidentity, split_struct = None):
    lret = []
    if split_struct is None:
        lrows = s.split(key_row)
        lrows = [row.split(key_col) for row in lrows]
        # print(lrows)
        for r in lrows:
            cell = [map_function(x) for x in r]
            lret.append(cell)
        if indsingles:
            lsingles = [row[0] for row in lret]
            lcells = [row[1:] if len(row) > 1 else row for row in lret]
            lret = (lsingles,lcells)
    else:
        lrows = str(s).split(key_row)
        r = 0
        lcells = []
        lsingles = []
        vlast = 1
        for row in lrows:
            row2 = row.split(key_col)
            row2 = [map_function(x) for x in row2]
            vlast = row2[-1]
            indstop = False
            while not indstop:
                if (r >= len(split_struct) # Too many cell values, ignore.
                or (len(row2) == 0 and len(split_struct) > 0)): # Cell exhausted.
                    indstop = True
                if not indstop:
                    if indsingles: # Singles split.
                        lsingles.append(row2[0]) # Row ratio.
                        if len(row2) > 1:
                            row2 = row2[1:]
                    if len(split_struct[r]) >= len(row2): # Repeat last value.
                        indstop = True
                        broadrow = row2 + [row2[-1]] * (len(split_struct[r]) - len(row2))
                        r = r + 1
                        lcells.append(broadrow)
                    else: # Overfilled this row, cut and move to next.
                        broadrow = row2[:len(split_struct[r])]
                        row2 = row2[len(split_struct[r]):]
                        r = r + 1
                        lcells.append(broadrow)
        # If not enough new rows, repeat the last one for entire base, preserving structure.
        cur = len(lcells)
        while cur < len(split_struct):
            lcells.append([vlast] * len(split_struct[cur]))
            cur = cur + 1
        lret = lcells
        if indsingles:
            lsingles = lsingles + [lsingles[-1]] * (len(split_struct) - len(lsingles))
            lret = (lsingles,lcells)
    return lret
    
def matrixdealer(self, split_ratio, baseratio):
    # print(split_ratio, baseratio)
    prompt = self.prompt
    if KEYBASE in prompt: prompt = prompt.split(KEYBASE,1)[1]
    if (KEYCOL in prompt.upper() or KEYROW in prompt.upper()):
        breaks = prompt.count(KEYROW) + prompt.count(KEYCOL) + int(self.usebase)
        # Prompt anchors, count breaks between special keywords.
        # print('prompt:', prompt)
        lbreaks = split_l2(prompt, KEYROW, KEYCOL, map_function = fcountbrk)
        # print('lbreaks', lbreaks)
        if (SPLROW not in split_ratio and (KEYROW in prompt.upper()) != (KEYCOL in prompt.upper())):
            # By popular demand, 1d integrated into 2d.
            # This works by either adding a single row value (inner),
            # or setting flip to the reverse (outer).
            # Only applies when using just ADDROW / ADDCOL keys, and commas in ratio.
            split_ratio = "1" + SPLCOL + split_ratio
            (split_ratio2r,split_ratio2) = split_l2(split_ratio, SPLROW, SPLCOL, indsingles = True,
                                map_function = ffloatd(1), split_struct = lbreaks)
        else: # Standard ratios, split to rows and cols.
            (split_ratio2r,split_ratio2) = split_l2(split_ratio, SPLROW, SPLCOL, indsingles = True,
                                            map_function = ffloatd(1), split_struct = lbreaks)
            print('split_ratio2r', split_ratio2r)
            print('split_ratio2', split_ratio2)
        # More like "bweights", applied per cell only.
        baseratio2 = split_l2(baseratio, SPLROW, SPLCOL, map_function = ffloatd(0), split_struct = lbreaks)
        print(baseratio2)
    (split_ratio,split_ratior) = ratiosdealer(split_ratio2,split_ratio2r)
    baseratio = baseratio2 
    
    # Merge various L2s to cells and rows.
    drows = []
    for r,_ in enumerate(lbreaks):
        dcells = []
        for c,_ in enumerate(lbreaks[r]):
            d = Region(split_ratio[r][c][0], split_ratio[r][c][1], baseratio[r][c], lbreaks[r][c])
            dcells.append(d)
        drow = Row(split_ratior[r][0], split_ratior[r][1], dcells)
        drows.append(drow)

    self.split_ratio = drows
    self.baseratio = baseratio

# class test:
#     def __init__(self, prompt,split_ratio=None,baseratio=0.2,usebase=False):
#         self.prompt = prompt
#         self.split_ratio = split_ratio
#         self.baseratio = 0.2
#         self.usebase = usebase
# test_prompt='a girl BREAK a cute boy BREAK a dog BREAK a tree.'
# split_ratio='1,1,1;1,1,1'
# x=test(test_prompt,split_ratio)
# keyconverter(x,split_ratio,usebase=False)
# print(x.prompt)
# matrixdealer(x, split_ratio, 0.2)

