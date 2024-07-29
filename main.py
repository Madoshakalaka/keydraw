#!/usr/bin/env python

"""
this draws an image of the split keyboard layout.
"""
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from typing import Optional
from typing import TypedDict, Self

from enum import Enum

class KeyXPosition(Enum):
    OuterMostColumn = 1
    PinkyColumn = 2
    RingColumn = 3
    MiddleColumn = 4
    IndexColumn = 5
    ThumbOuter = 6
    ThumbMiddle = 7
    ThumbInner = 8

class KeyYPosition(Enum):
    Top = 1
    SecondTop = 2
    ThirdTop = 3

class Side(Enum):
    Left = 1
    Right = 2


KEY_SIZE = 18.2

GAP_SIZE = 0.725

from dataclasses import dataclass


@dataclass
class Bounds:
    leftmost_x: float
    topmost_y: float
    rightmost_x: float
    bottommost_y: float


@dataclass
class RotatedSquare:
    lower_left: tuple[float, float]
    angle: float
    """
    0 to 90 degrees
    """
    def shrinked(self, scale: float) -> tuple[float, float, float, float, float, float, float, float]:
        """
        scale is a number between 0 and 1
        returns lower_left_x, lower_left_y, upper_left_x, upper_left_y, upper_right_x, upper_right_y, lower_right_x, lower_right_y
        """
        center = (self.lower_left[0] + np.sin(self.angle) * KEY_SIZE / 2, self.lower_left[1] - np.cos(self.angle) * KEY_SIZE / 2)
        return (center[0] * (1 - scale) + self.lower_left[0] * scale, center[1] * (1 - scale) + self.lower_left[1] * scale,
                center[0] * (1 - scale) + self.top_left()[0] * scale, center[1] * (1 - scale) + self.top_left()[1] * scale,
                center[0] * (1 - scale) + self.top_right()[0] * scale, center[1] * (1 - scale) + self.top_right()[1] * scale,
                center[0] * (1 - scale) + self.bottom_right()[0] * scale, center[1] * (1 - scale) + self.bottom_right()[1] * scale)

    @classmethod
    def ortho(cls, upper_right: tuple[float, float]) -> Self:
        return cls((upper_right[0], upper_right[1] + KEY_SIZE), 0.0)
    def bounds(self) -> Bounds:
        x, y = self.lower_left
        return Bounds(x, y - np.cos(self.angle) * KEY_SIZE, x + (1 + np.sin(self.angle)) * KEY_SIZE, y + np.sin(self.angle) * KEY_SIZE)
    def top_left(self) -> tuple[float, float]:
        return (self.lower_left[0] + np.sin(self.angle) * KEY_SIZE, self.lower_left[1] - np.cos(self.angle) * KEY_SIZE)
    def top_right(self) -> tuple[float, float]:
        return (self.lower_left[0] + (1 + np.sin(self.angle)) * KEY_SIZE, self.lower_left[1] - (np.cos(self.angle) - np.sin(self.angle)) * KEY_SIZE)
    def bottom_right(self) -> tuple[float, float]:
        return (self.lower_left[0] + np.cos(self.angle) * KEY_SIZE, self.lower_left[1] + np.sin(self.angle) * KEY_SIZE)



def thumb_coords(position: KeyXPosition, side: Side)-> RotatedSquare:

    outer_x = 83.5 - KEY_SIZE
    outer_y = 50.7 - KEY_SIZE
    match position:
        case KeyXPosition.ThumbOuter:
            # return RotatedSquare.ortho((outer_x, outer_y))
            return RotatedSquare.ortho((outer_x if side == Side.Left else -outer_x - KEY_SIZE, outer_y))
        case KeyXPosition.ThumbMiddle:
            angle = np.arcsin((40.5 - 1 - 2*KEY_SIZE) / KEY_SIZE)
            left = RotatedSquare((outer_x + KEY_SIZE + 1, outer_y + KEY_SIZE), angle)
            if side == Side.Left:
                return left
            else:
                return RotatedSquare((-left.bottom_right()[0], left.bottom_right()[1]), -angle)
        case KeyXPosition.ThumbInner:
            lower_left_x_increment = 40.5
            angle = np.arcsin((64.3 - lower_left_x_increment - KEY_SIZE) / KEY_SIZE)
            right = RotatedSquare((outer_x + lower_left_x_increment, outer_y + KEY_SIZE - 2), angle)
            if side == Side.Left:
                return right
            else:
                return RotatedSquare((-right.bottom_right()[0], right.bottom_right()[1]), -angle)


@dataclass
class Key:
    primary: Optional[str]
    right_pedal: Optional[str]
    both_pedal: Optional[str]
    left_pedal: Optional[str]
    position: tuple[KeyXPosition, Optional[KeyYPosition]]
    side: Side

    def square(self) -> RotatedSquare:
        match self.position[0]:
            case KeyXPosition.OuterMostColumn:
                x = 0.0
                match self.position[1]:
                    case KeyYPosition.Top:
                        y = 0.0
                    case KeyYPosition.SecondTop:
                        y = KEY_SIZE + GAP_SIZE
                return RotatedSquare.ortho((x if self.side == Side.Left else - x - KEY_SIZE, y))
            case KeyXPosition.PinkyColumn:
                x = KEY_SIZE + GAP_SIZE
                match self.position[1]:
                    case KeyYPosition.Top:
                        y = 0.0
                    case KeyYPosition.SecondTop:
                        y = KEY_SIZE + GAP_SIZE
                return RotatedSquare.ortho((x if self.side == Side.Left else - x - KEY_SIZE, y))
            case KeyXPosition.RingColumn:
                x = 2 * (KEY_SIZE + GAP_SIZE)
                base_y = -23.8
                match self.position[1]:
                    case KeyYPosition.Top:
                        y = base_y
                    case KeyYPosition.SecondTop:
                        y = base_y + KEY_SIZE + GAP_SIZE
                    case KeyYPosition.ThirdTop:
                        y = base_y + 2 * (KEY_SIZE + GAP_SIZE)
                return RotatedSquare.ortho((x if self.side == Side.Left else - x - KEY_SIZE, y))
            case KeyXPosition.MiddleColumn:
                
                x = 3 * (KEY_SIZE + GAP_SIZE)
                base_y = -23.8 - 2.7
                match self.position[1]:
                    case KeyYPosition.Top:
                        y = base_y
                    case KeyYPosition.SecondTop:
                        y = base_y + KEY_SIZE + GAP_SIZE
                    case KeyYPosition.ThirdTop:
                        y = base_y + 2 * (KEY_SIZE + GAP_SIZE)
                return RotatedSquare.ortho((x if self.side == Side.Left else - x - KEY_SIZE, y))
            case KeyXPosition.IndexColumn:
                x = 4 * (KEY_SIZE + GAP_SIZE)
                base_y = -4.0
                match self.position[1]:
                    case KeyYPosition.Top:
                        y = base_y
                    case KeyYPosition.SecondTop:
                        y = base_y + KEY_SIZE + GAP_SIZE
                return RotatedSquare.ortho((x if self.side == Side.Left else - x - KEY_SIZE, y))
            case KeyXPosition.ThumbOuter | KeyXPosition.ThumbMiddle | KeyXPosition.ThumbInner:
                return thumb_coords(self.position[0], self.side)

    


KEYS = [ 
        #    \     ^     @    |  
        Key("を", "ほ", "じょ", "きょ", (KeyXPosition.OuterMostColumn, KeyYPosition.Top), Side.Left),
        #           0          "
        Key("け", "わ", None, "ゎ",(KeyXPosition.OuterMostColumn, KeyYPosition.SecondTop), Side.Left),
        #           q Q A 
        Key("ち", "た", "きょ", "ちょ",(KeyXPosition.PinkyColumn, KeyYPosition.Top), Side.Left),
        Key("つ", "ぬ", "ず", "っ",(KeyXPosition.PinkyColumn, KeyYPosition.SecondTop), Side.Left),
        #           `     ~     W 
        Key("て", "ろ", "りょ", "てる",(KeyXPosition.RingColumn, KeyYPosition.Top), Side.Left),
        #           b    B      S   
        Key("と", "こ", "ご", "ど",(KeyXPosition.RingColumn, KeyYPosition.SecondTop), Side.Left),
        #           2   -      X
        Key("さ", "ふ", "ぶ", "しゃ",(KeyXPosition.RingColumn, KeyYPosition.ThirdTop), Side.Left),
        #          =     +     E
        Key("い", "へ", "れる", "ぃ",(KeyXPosition.MiddleColumn, KeyYPosition.Top), Side.Left),
        #           t T D
        Key("し", "か", "しゃ", "しょ",(KeyXPosition.MiddleColumn, KeyYPosition.SecondTop), Side.Left),
        Key("そ", "あ", "ぁ", "りょ",(KeyXPosition.MiddleColumn, KeyYPosition.ThirdTop), Side.Left),
        #          r    R     F
        Key("は", "す", "しゅ", "が",(KeyXPosition.IndexColumn, KeyYPosition.Top), Side.Left),
        #          4    $     V
        Key("ひ", "う", "ぅ", "び",(KeyXPosition.IndexColumn, KeyYPosition.SecondTop), Side.Left),
        Key(None, None, None, None, (KeyXPosition.ThumbOuter, None), Side.Left),
        Key(None, None, None, None, (KeyXPosition.ThumbMiddle, None), Side.Left),
        Key(None, None, None, None, (KeyXPosition.ThumbInner, None), Side.Left),
        #           u   U     J
        Key("ま", "な", "ない", "ます",(KeyXPosition.IndexColumn, KeyYPosition.Top), Side.Right),
        #           5 & M 
        Key("も", "え", "ぇ", "む",(KeyXPosition.IndexColumn, KeyYPosition.SecondTop), Side.Right),
        #           {    {    I
        Key("に", "゛", "「", "りょ",(KeyXPosition.MiddleColumn, KeyYPosition.Top), Side.Right),
        #           y Y K 
        Key("の", "ん", "じゅ", None,(KeyXPosition.MiddleColumn, KeyYPosition.SecondTop), Side.Right),
        #           6 ( <
        Key("ね", "お", "ぉ", "、",(KeyXPosition.MiddleColumn, KeyYPosition.ThirdTop), Side.Right),
        #           ] } O
        Key("ら", "゜", "」", "じゃ",(KeyXPosition.RingColumn, KeyYPosition.Top), Side.Right),
        #           g G L
        Key("り", "き", "ぎ", "だ",(KeyXPosition.RingColumn, KeyYPosition.SecondTop), Side.Right),
        # to do: assing wo somewhere
        #           7 )
        Key("る", "や", "ゃ", "。",(KeyXPosition.RingColumn, KeyYPosition.ThirdTop), Side.Right),
        #           p    P     N
        Key("み", "せ", "しゅ", "で",(KeyXPosition.PinkyColumn, KeyYPosition.Top), Side.Right),
        #           8    *     ?
        Key("め", "ゆ", "ゅ", "？",(KeyXPosition.PinkyColumn, KeyYPosition.SecondTop), Side.Right),
        #    ;     ±    °   :
        Key("れ", None, None, "べ",(KeyXPosition.OuterMostColumn, KeyYPosition.Top), Side.Right),
        
        
        Key(None, "よ", "ょ", None,(KeyXPosition.OuterMostColumn, KeyYPosition.SecondTop), Side.Right),

        Key(None, None, None, None, (KeyXPosition.ThumbOuter, None), Side.Right),
        #           h H 
        Key(None, "く", "げ", "ー", (KeyXPosition.ThumbMiddle, None), Side.Right),
        Key(None, None, None, None, (KeyXPosition.ThumbInner, None), Side.Right),
        
        
        
        


        ]

def is_small(k: str):
    """
    detect if a kana is small
    """
    return k in "ぁぃぅぇぉゃゅょっゎ"
def is_extended(k: str):
    return k in "ヵヶゐゑ"

def is_punctuation(k: str):
    return k in "、。・ー「」゛゜？"
def special_anchor(k: str) -> Optional[str]:
    if k == "「":
        return "mt"
    if k == "」":
        return "mm"
    if k == "゛":
        return "lt"
    if k == "゜":
        return "lt"
    return None


def get_bounds() -> (Bounds, Bounds):
    left_bounds = Bounds(0.0, 0.0, 0.0, 0.0)
    right_bounds = Bounds(0.0, 0.0, 0.0, 0.0)
    for key in KEYS:
        if key.side == Side.Right:
            key_bounds = key.square().bounds()
            right_bounds.leftmost_x = min(right_bounds.leftmost_x, key_bounds.leftmost_x)
            right_bounds.topmost_y = min(right_bounds.topmost_y, key_bounds.topmost_y)
            right_bounds.rightmost_x = max(right_bounds.rightmost_x, key_bounds.rightmost_x)
            right_bounds.bottommost_y = max(right_bounds.bottommost_y, key_bounds.bottommost_y)
        if key.side == Side.Left:
            key_bounds = key.square().bounds()
            left_bounds.leftmost_x = min(left_bounds.leftmost_x, key_bounds.leftmost_x)
            left_bounds.topmost_y = min(left_bounds.topmost_y, key_bounds.topmost_y)
            left_bounds.rightmost_x = max(left_bounds.rightmost_x, key_bounds.rightmost_x)
            left_bounds.bottommost_y = max(left_bounds.bottommost_y, key_bounds.bottommost_y)
    return ( left_bounds , right_bounds)

@dataclass
class Rect:
    top_left: tuple[float, float]
    bottom_right: tuple[float, float]

@dataclass
class ImageSpaceRect:
    lower_left: tuple[float, float]
    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_right: tuple[float, float]


    def symbol_coords(self, ) -> tuple[
            tuple[float, float]
            , tuple[float, float]
            , tuple[float, float]
            , tuple[float, float]
             ]:
        """
        each key is divided in to 3x3 grid, 

        1 2 3
        4 5 6
        7 8 9

        The primary symbol is drawn in the center of the rectangle composed of 4, 5, 6, 7, 8, 9.

        The right_pedal symbol is drawn in the center 3.

        The double_pedal symbol is drawn in the center 2.

        the left_pedal symbol is drawn in the center 1.
        """
        # cv2 anchors the text at the bottom left corner. The translation assumes that the symbol is a square.

        left_side_main_point = 1/2 * np.array(self.lower_left) + 1/2 * np.array(self.top_left)

        right_side_main_point = 1/2 * np.array(self.bottom_right) + 1/2 * np.array(self.top_right)

        main_symbol_point = left_side_main_point * 1/2 + right_side_main_point * 1/2 

        left_side_upper_point = 1/6 * np.array(self.lower_left) + 5/6 * np.array(self.top_left)
        right_side_upper_point = 1/6 * np.array(self.bottom_right) + 5/6 * np.array(self.top_right)
        right_pedal_point = left_side_upper_point * 1/6 + right_side_upper_point * 5/6
        both_pedal_point = left_side_upper_point * 1/2 + right_side_upper_point * 1/2
        left_pedal_point = left_side_upper_point * 5/6 + right_side_upper_point * 1/6



        return ((
            int(main_symbol_point[0]), int(main_symbol_point[1])

            ), 
            (int(right_pedal_point[0]), int(right_pedal_point[1])
            ),
            (int(both_pedal_point[0]), int(both_pedal_point[1])
            ),
            (int(left_pedal_point[0]), int(left_pedal_point[1])
            ))



MODIFIED = 'がぎぐげござじずぜぞだぢづでどびぶべぱぴぷぺ'

def draw_keyboards():
    """
    The image has two halves, left and right.
    Each half is shown in a rounded rectangle with blue border.
    """

    padding = 10
    ( left_bounds , right_bounds) = get_bounds()
    shrinkage = 0.9
    width = (left_bounds.rightmost_x - left_bounds.leftmost_x + 2 * padding) * shrinkage
    height = (left_bounds.bottommost_y - left_bounds.topmost_y + 2 * padding) * shrinkage

    

    pixel_per_mm = 6
    # Create a blank image with white background
    pil_image = Image.new('RGB', (297 * pixel_per_mm, 210 * pixel_per_mm), color='white')
    draw = ImageDraw.Draw(pil_image)

    # Calculate the center of the rectangles
    # such that the spacing around the rectangles is equal
    # (210 - 2*width) / 3 = spacing
    spacing = (297 - 2 * width) / 3
    assert spacing > 0
    center_left = (int(width / 2 + spacing), int(210 / 2))
    center_right = (int(297 - width / 2 - spacing), int(210 / 2))


    
    left_rect = Rect((center_left[0] - width / 2, center_left[1] - height / 2), (center_left[0] + width / 2, center_left[1] + height / 2))
    # Draw the left rectangle
    draw.rectangle([(int(pixel_per_mm*left_rect.top_left[0]), int(pixel_per_mm*left_rect.top_left[1])), (int(pixel_per_mm*left_rect.bottom_right[0]), int(pixel_per_mm*left_rect.bottom_right[1]))], outline=(255, 0, 0), width=2)

    right_rect = Rect((center_right[0] - width / 2, center_right[1] - height / 2), (center_right[0] + width / 2, center_right[1] + height / 2))
    # Draw the right rectangle
    draw.rectangle([(int(pixel_per_mm*right_rect.top_left[0]), int(pixel_per_mm*right_rect.top_left[1])), (int(pixel_per_mm*right_rect.bottom_right[0]), int(pixel_per_mm*right_rect.bottom_right[1]))], outline=(255, 0, 0), width=2)

    primary_font = ImageFont.truetype("./j.ttf", 24)
    right_font = ImageFont.truetype("./j.ttf", 19)
    both_font = ImageFont.truetype("./j.ttf", 19)
    left_font = ImageFont.truetype("./j.ttf", 19)
    cont_font = ImageFont.truetype("./j.ttf", 12)

    normal = []
    modified = []
    small = []
    bigram = []
    punct = []

    # Draw the keys
    for key in KEYS:

        def get_fill(symbol: str) -> tuple[int, int, int]:
            if symbol in MODIFIED:
                modified.append(symbol)
                return (0, 128, 0)
            if len(symbol) == 2:
                bigram.append(symbol)
                # purple
                return (36, 36, 96)

            if is_small(symbol):
                small.append(symbol)
                # orange
                return (255, 165, 0)
            elif is_extended(symbol):
                # gray
                return (200, 200, 200)
            elif is_punctuation(symbol):
                punct.append(symbol)
                # red
                return (128, 0, 0)
            else:
                normal.append(symbol)
                return (0, 0, 0)

        square = key.square()
        lower_left_x, lower_left_y, top_left_x, top_left_y, top_right_x, top_right_y, bottom_right_x, bottom_right_y = square.shrinked(0.9)
        if key.side == Side.Right:
            center = center_right
            bounds = right_bounds
        if key.side == Side.Left:
            center = center_left
            bounds = left_bounds


        lower_left_x = ( lower_left_x - bounds.leftmost_x ) * shrinkage + padding + center[0] - width / 2
        lower_left_y = ( lower_left_y - bounds.topmost_y )*shrinkage + padding + center[1] - height / 2
        top_left_x = ( top_left_x - bounds.leftmost_x )*shrinkage + padding + center[0] - width / 2
        top_left_y = ( top_left_y - bounds.topmost_y )*shrinkage + padding + center[1] - height / 2

        top_right_x = ( top_right_x - bounds.leftmost_x )*shrinkage + padding + center[0] - width / 2
        top_right_y = ( top_right_y - bounds.topmost_y )*shrinkage + padding + center[1] - height / 2

        bottom_right_x = ( bottom_right_x - bounds.leftmost_x )*shrinkage + padding + center[0] - width / 2
        bottom_right_y = ( bottom_right_y - bounds.topmost_y )*shrinkage + padding + center[1] - height / 2


        rect = ImageSpaceRect((int(pixel_per_mm*lower_left_x), int(pixel_per_mm*lower_left_y)), (int(pixel_per_mm*top_left_x), int(pixel_per_mm*top_left_y)), (int(pixel_per_mm*top_right_x), int(pixel_per_mm*top_right_y)), (int(pixel_per_mm*bottom_right_x), int(pixel_per_mm*bottom_right_y))
        )
        draw.line([rect.lower_left, rect.top_left], fill=(0, 0, 0), width=4)
        draw.line([rect.top_left, rect.top_right], fill=(0, 0, 0), width=4)
        draw.line([rect.top_right, rect.bottom_right], fill=(0, 0, 0), width=4)
        draw.line([rect.bottom_right, rect.lower_left], fill=(0, 0, 0), width=4)



        main_symbol, right_pedal_symbol, both_pedal_symbol, left_pedal_symbol = rect.symbol_coords()
        if key.side == Side.Right:
            print(key.primary, main_symbol)



        if key.primary is not None:
            draw.text(main_symbol, key.primary, font=primary_font, fill=get_fill(key.primary), anchor="mm")
        if key.right_pedal is not None:
            anchor = special_anchor(key.right_pedal)
            if anchor is None:
                anchor = "mm"
            draw.text(right_pedal_symbol, key.right_pedal, font=right_font, fill=get_fill(key.right_pedal), anchor=anchor)
        if key.both_pedal is not None:
            anchor = special_anchor(key.both_pedal)
            if anchor is None:
                anchor = "mm"
            draw.text(both_pedal_symbol, key.both_pedal, font=cont_font if len(key.both_pedal) == 2 else both_font, fill=get_fill(key.both_pedal), anchor=anchor)
        if key.left_pedal is not None:
            anchor = special_anchor(key.left_pedal)
            if anchor is None:
                anchor = "mm"
            draw.text(left_pedal_symbol, key.left_pedal, font=cont_font if len(key.left_pedal) == 2 else left_font, fill=get_fill(key.left_pedal), anchor=anchor)


            # cv2.putText(img, key.primary, main_symbol, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    # Convert the PIL image to Numpy array and switch from RGB to BGR
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # print 
    print("normal")
    for k in normal:
        print(k)
    print()
    print("modified")
    for k in modified:
        print(k)
    print()
    print("small")
    for k in small:
        print(k)
    print()
    print("bigram")
    for k in bigram:
        print(k)
    print()
    print("punct")
    for k in punct:
        print(k)
    print()


            


    # Display the image
    cv2.imshow('Keyboard Layout', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    draw_keyboards()





    


