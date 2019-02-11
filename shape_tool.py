import tkinter as tk
from PIL import Image, ImageTk, ExifTags
import json

class Point2D:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.x) + " " + str(self.y)

class Polygon:
    def __init__(self):
        self.vertices = []
        self._lines = []
    def add(self, x, y, w):
        l = len(self.vertices)
        if  l >= 4:
            return
        self.vertices.append(Point2D(x, y))
        l += 1
        if l >= 2:
            self._lines.append(w.create_line(self.vertices[l - 2].x, self.vertices[l - 2].y,
                                             self.vertices[l - 1].x, self.vertices[l - 1].y, fill="blue"))
        if l == 4:
            self._lines.append(w.create_line(self.vertices[l - 1].x, self.vertices[l - 1].y,
                                             self.vertices[0].x, self.vertices[0].y, fill="blue"))

    def dump(self):
        result = []
        for i in self.vertices:
            result.append(i.x)
            result.append(i.y)
        return result

'''def drag_motion(event):
    x = self.winfo_x() - self.drag_start_x + event.x
    y = self.winfo_y() - self.drag_start_y + event.y
    self.place(x=x, y=y)'''

# --- main ---

# init
class App():
    def __init__(self):
        self.root = tk.Tk()
        self.w = tk.Canvas(self.root, width=100, height=100)
        self.poly = None
        self.current_path = None
        self.image = None
        self.data = {}
        self.wimage = None

        self.w.bind('<Button-1>', self.on_click)

        self.bv = tk.Button(self.root, text="Validate", command=self.validate)
        self.bv.pack()

        self.bn = tk.Button(self.root, text="Next", command=self.next)
        self.bn.pack()

    def next(self):
        self.load_image(path="data/ndf_crop.jpg")

    def load_image(self, path="data/ndf.jpg"):
        self.current_path = path
        self.poly = Polygon()
        self.image = Image.open(path)

        try:
            orientation = ""
            for i in ExifTags.TAGS.keys():
                if ExifTags.TAGS[i] == 'Orientation':
                    orientation = i
                    break
            exif = dict(self.image._getexif().items())

            if exif[orientation] == 3:
                self.image = self.image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                self.image = self.image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                self.image = self.image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            pass
        ratio = self.downscale_image(self.image)
        maxsize = (self.image.width // ratio, self.image.height // ratio)
        self.image.thumbnail(maxsize, Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.image)
        self.w.config(width=self.image.width, height=self.image.height)
        self.wimage = self.w.create_image(0,0, anchor=tk.NW, image=self.photo)
        self.w.pack()


    def save(self):
        with open('data/data.json', 'w') as outfile:
            json.dump(self.data, outfile)


    def validate(self):
        self.data[self.current_path] = {
            "dimensions": [self.image.width, self.image.height],
            "poly": self.poly.dump()
        }
        print(self.data)
        self.save()

    def downscale_image(self, im, max_dim=800):
        """Shrink im until its longest dimension is <= max_dim.
        Returns new_image, scale (where scale <= 1).
        """
        a, b = im.width, im.height
        if max(a, b) <= max_dim:
            return 1.0

        ratio = max(a, b) / max_dim
        return ratio

    def on_click(self, event=None):
        # `command=` calls function without argument
        # `bind` calls function with one argument
        self.poly.add(event.x, event.y, self.w)

app = App()

with open('data/data.json') as json_file:
     app.data = json.load(json_file)

# "start the engine"
app.load_image()
app.root.mainloop()