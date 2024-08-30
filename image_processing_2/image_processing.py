#!/usr/bin/env python3


import math
from PIL import Image

def get_pixel(image, row, col):
    """Gets value of pixel at coordinate (col, row)
    image: dict,
    row: int,
    col: int,
    Returns an integer value
    """
    # set variables to attributes of the image
    height, width, pixels = image["height"], image["width"], image["pixels"]

    # get the index of the pixel to be output. Rows --> height, Columns --> width
    loc = width * row + col  # counting from 0 for all parameters

    # can't work with None. Change it to a dummy list
    if pixels is None:
        image = {"height": height, "width": width, "pixels": [0] * (height * width)}

    return pixels[loc]


def advanced_get_pixel(image, row, col, boundary_behavior):
    """Takes in an image object in the form of a dictionary.
    image: dict,
    row: int,
    col: int,
    boundary_behavior: int: accounts for handling edge effects,
    Returns the pixel at given coordinate
    """

    # new variables for ease of reference
    pixels, height, width = image["pixels"], image["height"], image["width"]

    # the index/location to get the pixel from
    loc = row * width + col

    # return a pixel according to specified behavior

    # case 1: set non-existent values to 0
    if boundary_behavior == "zero":

        # checking whether or not the index specified is within bounds
        # out of bounds, set to 0

        pixel = 0

        if (0 <= row <= height - 1) and (0 <= col <= width - 1):
            # within bounds so get actual value
            pixel = pixels[loc]

    # case 2: extending by taking values in the outmost edge
    if boundary_behavior == "extend":

        # if the column/row is negative or if the column/row is > width/height - 1
        col = max(min(col, width - 1), 0)
        row = max(min(row, height - 1), 0)

        # update loc
        loc = row * width + col
        # get pixel at that index
        pixel = pixels[loc]

    # case 3: wrapping around by taking values from opposite edge and same row
    if boundary_behavior == "wrap":

        row = row % height
        col = col % width

        # update loc
        loc = row * width + col

        # get the pixel at that index
        pixel = pixels[loc]

    # central return for all behaviors
    return pixel


def set_pixel(image, row, col, color):
    """Sets the pixel value for an image at coordinate (col, row) to color
    image: dict,
    row: int,
    col: int,
    color: int,
    Returns None (implicitly)
    """
    # set variables to attributes of the image
    width = image["width"]
    loc = width * row + col  # assuming counting from 0 for all parameters
    image["pixels"][loc] = color


def apply_per_pixel(image, func):
    """Applies a filter, func, to a given image object
    image: dict,
    func: filter function
    Returns filtered image"""

    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": image["pixels"][
            :
        ],  # changed the pixels from empty list to the original image pixels, as a copy
    }
    for row in range(image["height"]):
        for col in range(image["width"]):
            color = get_pixel(image, row, col)  # switched order of col and row
            new_color = func(
                color
            )  # definition of inverted takes in an image(dict). Color? List?
            set_pixel(
                result, row, col, new_color
            )  # indented inwards by 1 tab. Same as before but clearer
    return result


def inverted(image):
    """Creates an inverted image of the original
    image: dict,
    Returns filtered image where each pixel is 255 - original"""
    return apply_per_pixel(
        image, lambda color: 255 - color
    )  # the range is [0, 255] not 256


# HELPER FUNCTIONS


def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    Take kernel as a list going in left - right order.
    Pixels adjacent to pixel in question will also be a list in the same order.
    Corresponding indices have their values multiplied together.
    """

    # check if boundary behavior is specified. Otherwise return None
    if boundary_behavior not in ["zero", "extend", "wrap"]:
        return

    # loop over the number of rows
    # loop over the number of columns
    # consider each pixel. Take values surrounding it.
    # get the width of one side of the kernel
    kernel_width = len(kernel) ** 0.5

    # iterate from -ve to +ve but still in kernel_iterator range
    kernel_iterator = int((kernel_width - 1) / 2)
    # variable to store the pixels adjacent to given pixel

    # assuming kernel is a list in row-major order (left - right)
    # create new image object in case of manipulating values
    new_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [0] * image["height"] * image["width"],
    }

    # new variables for ease of reference
    height, width = (
        new_image["height"],
        new_image["width"],
    )

    for row in range(height):  # gets to height - 1
        for col in range(width):  # gets to width - 1
            # extract the pixel at a given index of the pixels array.

            # manipulating the pixel to give a new value using the kernel

            # go over the mini-rows

            ####temp = 0
            temp = 0

            for r in range(
                -kernel_iterator, kernel_iterator + 1
            ):  # controlling for exclusive nature of range fn
                # go over the mini-columns
                for c in range(-kernel_iterator, kernel_iterator + 1):

                    kernel_index = int(
                        (r + kernel_iterator) * kernel_width + (c + kernel_iterator)
                    )

                    adj_pix = advanced_get_pixel(
                        image, row + r, col + c, boundary_behavior
                    )

                    ####new method. Alternative to zip below

                    temp += kernel[kernel_index] * adj_pix

            set_pixel(new_image, row, col, temp)  ####changed new_pixel_value to temp

            # reset the pixel container for next iteration
            temp = 0

    return new_image


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """

    # get the pixels and loop over all of them. Function will not return anything
    pixels = image["pixels"]
    output_pixels = []
    for pixel in pixels:
        # for non-int pixels

        ###if type(pixel) != int: round(pixel)

        # for every pixel
        pixel = round(max((min(pixel, 255), 0)))
        output_pixels.append(pixel)

    output_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": output_pixels,
    }
    return output_image


def make_kernel(kernel_size):
    return [1 / kernel_size**2] * kernel_size**2


# FILTERS


def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)

    # then compute the correlation of the input image with that kernel
    blurred_image = correlate(image, make_kernel(kernel_size), "extend")

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.

    # rounding and clipping the image so it is in range
    output_image = round_and_clip_image(blurred_image)

    return output_image


def sharpened(image, n):
    """Returns an image obtained by sharpening
    given an image of type dict and a kernel size n.
    Should not mutate the input image.
    """

    # finding a single correlation using identity matrix/kernel
    one_index = int((n * n - 1) / 2)
    two_kernel = [0] * n**2
    two_kernel[one_index] = 2  # the outer two factors everything by 2

    # blur kernel

    blur_kernel = make_kernel(n)

    final_kernel = [a - b for a, b in zip(two_kernel, blur_kernel)]

    output_image = correlate(image, final_kernel, "extend")
    output_image = round_and_clip_image(output_image)

    return output_image


def edges(image):
    """Returns a copy of the input image where the edges are emphasized.
    Should not alter the input image."""

    k1 = [-1, -2, -1, 0, 0, 0, 1, 2, 1]

    k2 = [-1, 0, 1, -2, 0, 2, -1, 0, 1]

    im_k1 = correlate(image, k1, "extend")
    im_k2 = correlate(image, k2, "extend")

    pixels1, pixels2 = im_k1["pixels"], im_k2["pixels"]

    output_pixels = [math.sqrt((a**2 + b**2)) for a, b in zip(pixels1, pixels2)]
    output_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": output_pixels,
    }

    return round_and_clip_image(output_image)


# VARIOUS FILTERS


# Helper function to split image into three different ones based on r, g, b pixel values


def split_image(image):
    """
    Takes in a color image and splits it into 3
    different greyscale images based on r, g, b values.
    Returns three greyscale versions of image.
    """

    pixels = image["pixels"]

    # new image from the red pixels
    red_image = {"height": image["height"], "width": image["width"], "pixels": []}

    # new image from the green pixels
    green_image = {"height": image["height"], "width": image["width"], "pixels": []}

    # new image from the blue pixels
    blue_image = {"height": image["height"], "width": image["width"], "pixels": []}

    for pixel in pixels:
        red_image["pixels"] += [(pixel[0])]
        green_image["pixels"] += [(pixel[1])]
        blue_image["pixels"] += [(pixel[2])]

    return red_image, green_image, blue_image


def combine_images(image1, image2, image3):
    """
    Given 3 greyscale images, combines the pixels.
    Assumes image1 has the red pixels,
    image2 has the green and image 3 has the blue pixels.
    Returns a new color image
    """

    # initializing new color image object
    image = {"height": image1["height"], "width": image2["width"], "pixels": []}

    for i in range(image1["height"] * image1["width"]):
        pixels = (image1["pixels"][i], image2["pixels"][i], image3["pixels"][i])
        image["pixels"] += [(pixels)]

    return image


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def applying_filter(image):
        """
        Takes in a color image and returns filtered versions.
        """

        # split the input image into 3 greyscale ones according tho the r, g, b pixels
        split_images = split_image(image)

        # variable to hold filtered greyscale images
        filtered_images = []
        for new_image in split_images:
            # applying the filter to each image in the set of split images
            new_image = filt(new_image)
            filtered_images.append(new_image)

        # combine the images together to make a color image

        return combine_images(
            filtered_images[0], filtered_images[1], filtered_images[2]
        )

    return applying_filter


def make_blur_filter(kernel_size):
    """
    Makes a blurry filter in a way that is consistent with
    color_filter_from_greyscale_filter so it can be called with a single argument
    """

    def filtering(image):
        """Takes an image and blurs it."""

        new_image = blurred(image, kernel_size)
        return new_image

    return filtering


def make_sharpen_filter(kernel_size):
    """
    Makes a sharpening filter in a way that is consistent with
    color_filter_from_greyscale_filter so it can be called with a single argument
    """

    def filtering(image):
        """Takes an image and sharpens it."""

        new_image = sharpened(image, kernel_size)
        return new_image

    return filtering


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """

    def apply_filters(image):
        """
        Takes in an image and checks whether it is a greyscale image or a color image.
        If it is a color image, it uses color_filter_from_greyscale_filter
        to apply the filters to the image.
        If it is a greyscale image, it just applies the filters normally.
        Returns the filtered image.
        """

        for a_filter in filters:
            image = a_filter(image)

        return image

    return apply_filters


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """

    width, height, pixels = image["width"], image["height"], image["pixels"][:]
    new_image = {"width": width, "height": height, "pixels": pixels}

    i = 0
    while i < ncols:
        grey = greyscale_image_from_color_image(new_image)
        energy = compute_energy(grey)
        c_energy_map = cumulative_energy_map(energy)
        seam = minimum_energy_seam(c_energy_map)
        new_image = image_without_seam(new_image, seam)

        i += 1

    return new_image


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    # new image object
    greyscale_image = {"height": image["height"], "width": image["width"], "pixels": []}

    pixels = greyscale_image["pixels"]

    for a_pixel in image["pixels"]:
        red, green, blue = a_pixel[0], a_pixel[1], a_pixel[2]
        new_pixel = round(0.299 * red + 0.587 * green + 0.114 * blue)
        pixels.append(new_pixel)

    return greyscale_image


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """

    width = energy["width"]
    height = energy["height"]

    # fill the top row using the first row of energy
    cumm_energies = energy["pixels"][:]

    # now you start from the energy['width']-th index onwards.
    # Take the value at that index and the min of
    # (index-width, index-width+1, index-width+1)

    for index in range(width, width * height):

        # check for the values that are at the ends so have two adjacent values

        # case 1: the pixel is at the right hand end
        if index % width == width - 1:
            value = cumm_energies[index] + min(
                cumm_energies[index - width - 1], cumm_energies[index - width]
            )

        # case 2: the pixel is at the left hand end
        if index % width == 0:
            value = cumm_energies[index] + min(
                cumm_energies[index - width], cumm_energies[index - width + 1]
            )

        # case 3: the pixel is somewhere in the middle
        if (index % width != 0) and (
            index % width != width - 1
        ):  # did not work with else statement?

            value = cumm_energies[index] + min(
                cumm_energies[index - width - 1],
                cumm_energies[index - width],
                cumm_energies[index - width + 1],
            )

        cumm_energies[index] = value

    energy_map = {"height": height, "width": width, "pixels": cumm_energies}

    return energy_map


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """

    # cem values
    cem_values = cem["pixels"]
    # width and height
    width = cem["width"]
    height = cem["height"]
    # get the value of the least bottom pixel
    bottom_pixel = min(cem_values[-width:])

    # starting index
    start_index = 0

    # locate the index of the bottom pixel

    for i in range(
        (height * width - width), (height * width)
    ):  # exclusive so goes to the end

        if cem_values[i] == bottom_pixel:
            start_index = i

            break

    minimum_path = []

    # backtracking from the least index to the top
    for _ in range(height):
        minimum_path.append(start_index)

        center = start_index - width
        left = start_index - width - 1
        right = start_index - width + 1

        # check least value and account for values from the ends
        if start_index % width == 0:

            start_index = center
            if cem_values[right] < cem_values[center]:
                start_index = right

        elif start_index % width == width - 1:

            start_index = left
            if cem_values[center] < cem_values[left]:
                start_index = center

        else:

            # checks and balances to make sure that the correct index is taken
            start_index = left
            if cem_values[right] < cem_values[left]:
                start_index = right
            if cem_values[center] < cem_values[left]:
                start_index = center
                if cem_values[right] < cem_values[center]:
                    start_index = right

    return minimum_path


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """

    pixels = image["pixels"][:]

    for ix in sorted(seam, reverse=True):
        del pixels[ix]

    width = image["width"] - 1
    height = image["height"]

    new_image = {"width": width, "height": height, "pixels": pixels}

    return new_image



def custom(image):
    pixels = []
    width = image["width"]
    height = image["height"]    
    
    
    for i in range(width*height-1):
        new_pixels = []
        for j in range(3):
            
            if image['pixels'][i][j] < 128:
                new_pixels.append(255 - image['pixels'][i][j])
            else: new_pixels.append(image['pixels'][i][j])
        pixels.append(tuple(new_pixels))

   
    new_image = {"width": width, "height": height, "pixels": pixels}

    return new_image

        



def custom_feature(image):
    """Takes in an image object and returns a new image
    that looks like it has been blurred and made 'floaty'
    to the eye.
    image: dict object
    """

    # custom kernel that creates effect
    kernel = [1, -1, 1, 0, -1, 0, 1, -1, 1]

    # variable to hold the split image in R, G, B format
    sub_images = split_image(image)
    new_subs = []

    # apply matrix on RGB components of image
    for im in sub_images:

        new_subs.append(correlate(im, kernel, "extend"))

    return combine_images(new_subs[0], new_subs[1], new_subs[2])


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    # im1 = load_color_image("test_images/cat.png")
    # new_im1 = color_filter_from_greyscale_filter(inverted)(im1)
    # save_color_image(new_im1, "new_cat.png")

    # im2 = load_color_image('test_images/python.png')
    # im3 = load_color_image('test_images/sparrowchick.png')
    # new_im2 = color_filter_from_greyscale_filter(make_blur_filter(9))(im2)
    # new_im3 = color_filter_from_greyscale_filter(make_blur_filter(9))(im3)
    # save_color_image(new_im2, 'created_images/blurred_python.png')
    # save_color_image(new_im3, 'created_images/sharpened_sparrowchick.png')

    # filter1 = color_filter_from_greyscale_filter(edges)
    # filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    # filt = filter_cascade([filter1, filter1, filter2, filter1])
    # im4 = load_color_image("test_images/frog.png")
    # save_color_image(filt(im4), "created_images/frog_filter_cascade.png")

    # two_cats = load_color_image('test_images/twocats.png')
    # carved_cats = seam_carving(two_cats, 100)
    # save_color_image(carved_cats, 'created_images/carved_cats.png')

    # pattern = load_color_image('test_images/pattern.png')
    # grey = greyscale_image_from_color_image(pattern)
    # save_greyscale_image(grey, 'greypattern.png')
    # pattern_energy_map = compute_energy(grey)
    # save_greyscale_image(pattern_energy_map, 'pattern_energy_map.png')
    # print(f"\nPattern energies are: {pattern_energy_map}\n")
    # pattern_cummulative = cumulative_energy_map(pattern_energy_map)
    # save_greyscale_image(pattern_cummulative, 'pattern_cummulative.png')
    # print(f"\nThe cummulative energy map is: {pattern_cummulative}\n")
    # min_seam = minimum_energy_seam(pattern_cummulative)
    # print(f'This is the seam: {min_seam}')
    # save_color_image(image_without_seam(pattern, min_seam), 'finalpattern.png')

    # kernel = [0, 0, -1, 
    #           0, 0, 0, 
    #           1, 0, 0]
    
    # subs = list(split_image(im))
    # for image in subs:
    #     subs.append((correlate(image, kernel, 'wrap')))
    # subs = subs[3:]
    # im = combine_images(subs[0], subs[1], subs[2])
    # save_color_image(im, 'tryout.png')


    # new_image = custom_feature(im)
    # save_color_image(new_image, 'created_images/custom_feature.png')
    # blur = color_filter_from_greyscale_filter(make_blur_filter(3))(im)
    # save_color_image(blur, 'comp.png')


    #March 11
    img1 = load_color_image('random_images/square.png')
    # img1 = color_filter_from_greyscale_filter(inverted)(img1)
    # save_color_image(img1, 'random_images/square_inverted.png')
    # img2 = color_filter_from_greyscale_filter(make_sharpen_filter(100))(img1)
    # save_color_image(img2, 'random_images/square_sharpen2.png')

    filter1 = color_filter_from_greyscale_filter(edges)
    save_color_image(filter1(img1), "random_images/square_filter_cascade.png")

    pass
