import cv2
import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import numpy as np

class LabImageProcessing:
    """
    A class used to bind OpenCV for Image Processing Lab.

    This class is a container for various subclasses that perform different image processing
    operations. It includes subclasses for opening images, visualizing them, applying various
    image processing tools, conducting morphological manipulations, and implementing thresholding techniques.

    Subclasses
    ----------
    ImageOpener : class
        Used to open images from a file system and manage image data.
    ImageVisualization : class
        Offers functionalities for visualizing images and their characteristics.
    ImageTools : class
        Provides tools for image processing operations like resizing, channel extraction, and ROI definition.
    MorphologicalManipulation : class
        Allows application of morphological operations such as erosion and dilation.
    Thresholding : class
        Implements various thresholding techniques on image channels.
    Examples : class
        A class for demonstrating example use cases of plot and image vizualisation. (Only Demo)

    Notes
    -----
    Each subclass is equipped with its own set of attributes and methods tailored to its specific purpose
    in image processing.

    
    """
    # Class definitions as provided...

    class ImageOpener:        
        """
        A class used to open images from a file system.

        This class provides functionalities to open an image file from a specified path 
        and access its data in the form of a numpy array. If no path is provided, it 
        opens a file dialog allowing the user to select an image file. The class can 
        also return the path of the currently opened image.

        Attributes
        ----------
        image_path : str
            Path of the image file. Initially set to an empty string.
        image : ndarray or None
            Image data in numpy array format. Initialized as None, and populated with 
            image data after an image is opened.

        Methods
        -------
        open_image(image_path=None)
            Opens an image file from the given path or a file dialog if no path is provided.
        get_image()
            Returns the current image as a numpy array.
        get_image_path()
            Returns the path of the current image as a string.
        """

        def __init__(self):

            """
            Initializes the ImageOpener class with default values for attributes.
            
            Sets the `image_path` attribute to an empty string and `image` attribute to None.
            """
            self.image_path = ""
            self.image = None

        def open_image(self, image_path=None):

            """
            Opens an image file from the specified path or via a file dialog.

            If `image_path` is not provided, this method opens a file dialog for the user
            to select an image file. The selected image is then read into a numpy array
            and stored in the `image` attribute. The path of the image is stored in the 
            `image_path` attribute.

            Parameters
            ----------
            image_path : str, optional
                The path of the image to be opened. If None, a file dialog is opened for 
                image selection (default is None).

            Returns
            -------
            None
            """
            if image_path is None:
                root = tk.Tk()
                root.withdraw()
                self.image_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~/Pictures"))
            else:
                self.image_path = image_path
            if self.image_path:
                self.image = cv2.imread(self.image_path)
        
        def get_image(self):
            """
            Returns the current image.

            Retrieves the image data stored in the `image` attribute.

            Returns
            -------
            ndarray
                The current image as a numpy array.
            """
            return self.image

        def get_image_path(self):
            """
            Returns the path of the current image.

            Retrieves the path of the image stored in the `image_path` attribute.

            Returns
            -------
            str
                The path of the current image.
            """
            return self.image_path


    class ImageVisualization:
        """
        A class for visualizing various aspects of images using OpenCV and Matplotlib.

        This class offers functionalities for visualizing different channels of an image, 
        plotting histograms, displaying images, and showing images with contours. It can 
        handle both grayscale and color images and provides methods for specific visual 
        analyses such as identifying peak bounds in a histogram.

        Parameters
        ----------
        image : ndarray
            The image to be visualized.

        Methods
        -------
        plot_all_image_channels()
            Plots RGB and HSV channels of the current image.
        plot_histogram(channel_data, grid=False, xmin=None, xmax=None)
            Plots a histogram for a given channel of the image.
        plot_image(image)
            Displays the given image using Matplotlib.
        show_image(image=None, title="Image")
            Displays the given or current image using OpenCV.
        show_image_with_contour(image, contour)
            Displays the given image with a specified contour.
        get_image_with_contour(image, contour)
            Returns a copy of the image with the specified contour.
        plot_peak_bounds(channel)
            Plots histogram of a channel and highlights the peak bounds.
        """
        def __init__(self, image):
            """
            Initializes the ImageVisualization class with an image.

            Parameters
            ----------
            image : ndarray
                The image to be visualized.
            """
            self.image = image

        def plot_all_image_channels(self):        
            """
            Plots individual RGB and HSV channels of the current image.

            This method converts the current image from BGR to RGB and BGR to HSV,
            then splits these into their respective channels (Red, Green, Blue, Hue,
            Saturation, and Value). Each channel is then plotted separately using 
            Matplotlib. If no image is currently loaded, an error message is printed.

            Returns
            -------
            tuple of (None, str) or None
                If the image is not loaded, returns (None, "No image"). Otherwise, 
                returns None after displaying the plots.
            """
            if self.image is None:
                print("Error: Could not read the image.")
                return None, "No image"

            # Convert image from BGR to RGB
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # Convert image from BGR to HSV
            image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

            # Split into individual channels
            r, g, b = cv2.split(image_rgb)
            h, s, v = cv2.split(image_hsv)

            # Plotting the channels
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))

            axs[0, 0].imshow(r, cmap='Reds')
            axs[0, 0].set_title('Red Channel')
            axs[0, 0].axis('off')

            axs[0, 1].imshow(g, cmap='Greens')
            axs[0, 1].set_title('Green Channel')
            axs[0, 1].axis('off')

            axs[0, 2].imshow(b, cmap='Blues')
            axs[0, 2].set_title('Blue Channel')
            axs[0, 2].axis('off')

            axs[1, 0].imshow(h, cmap='hsv')
            axs[1, 0].set_title('Hue Channel')
            axs[1, 0].axis('off')

            axs[1, 1].imshow(s, cmap='gray')
            axs[1, 1].set_title('Saturation Channel')
            axs[1, 1].axis('off')

            axs[1, 2].imshow(v, cmap='gray')
            axs[1, 2].set_title('Value Channel')
            axs[1, 2].axis('off')

            plt.show()

        def plot_histogram(self, channel_data, grid=False, xmin=None, xmax=None):
            """
            Plots a histogram for a given channel of the image.

            This method calculates and plots the histogram of the provided channel data.
            The histogram shows the distribution of pixel intensity values. Options to 
            add a grid and set the range of x-axis (intensity values) are available.

            Parameters
            ----------
            channel_data : ndarray
                The channel data (e.g., a color or grayscale channel) for which the 
                histogram is to be plotted.
            grid : bool, optional
                Whether to display a grid on the plot (default is False).
            xmin : int, optional
                The minimum intensity value on the x-axis (default is 0).
            xmax : int, optional
                The maximum intensity value on the x-axis (default is 256).

            Returns
            -------
            None
            """
            # Calculate the histogram
            hist = cv2.calcHist([channel_data], [0], None, [256], [0, 256])

            # Plot the histogram
            plt.figure(figsize=(10, 6))
            plt.plot(hist, color='gray')
            plt.title('Mono Chromatic Histogram')
            plt.xlabel('Intensity Value')
            plt.ylabel('Pixel Count')
            
            if xmin is None:
                xmin = 0
            if xmax is None:
                xmax = 256
            # Set x-axis limits if valid xmin and xmax are provided
            if xmin is not None and xmax is not None and 0 <= xmin < xmax <= 256:
                plt.xlim([xmin, xmax])

            # Add grid if requested
            if grid:
                plt.grid(True)

            plt.show()


        def plot_image(self, image): 
            """
            Displays the given image using Matplotlib.

            This method checks if the image is grayscale and, if so, converts it to RGB
            format before displaying. It uses Matplotlib to plot the image with axis 
            labels and ticks turned off.

            Parameters
            ----------
            image : ndarray
                The image to be displayed.

            Returns
            -------
            None
            """
            # Check if the image is grayscale
            if len(image.shape) == 2:
                # Convert grayscale image to RGB format
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                # If it's already in RGB format, keep it as is
                image_rgb = image

            # Display the image using Matplotlib
            plt.figure(figsize=(8, 8))
            plt.imshow(image_rgb)
            plt.axis('off')  # Turn off axis labels and ticks
            plt.show()
        
        def show_image(self, image=None, title="Image"):
            """
            Displays the given or current image using OpenCV.

            If no image is provided, the method uses the current image stored in the class.
            The image is displayed in a window with the specified title. The window remains 
            open until the 'Q' key is pressed.

            Parameters
            ----------
            image : ndarray, optional
                The image to be displayed. If None, the current image is used (default is None).
            title : str, optional
                The title of the window in which the image is displayed (default is "Image").

            Returns
            -------
            None
            """
            if image is None:
                image = self.image
            cv2.imshow(f"{title} (Press Q to close this)", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def show_image_with_contour(self, image, contour):
                """
                Displays the given image with a specified contour using OpenCV.

                This method first calls `get_image_with_contour` to apply the contour to the image
                and then displays the result. The window title indicates that it is showing an 
                image with a contour.

                Parameters
                ----------
                image : ndarray
                    The image to which the contour will be applied.
                contour : ndarray
                    The contour to be applied to the image.

                Returns
                -------
                None
                """
                # Merge the channels back
                self.show_image(self.get_image_with_contour(image, contour), title="Image with Contour")

        def get_image_with_contour(self, image, contour):
                """
                Returns a copy of the image with the specified contour applied.

                The contour is applied by setting the pixels within the contour to a specific 
                color (in this case, red). This method splits the image into its color channels, 
                applies the contour, and then merges the channels back.

                Parameters
                ----------
                image : ndarray
                    The original image.
                contour : ndarray
                    The contour mask to be applied.

                Returns
                -------
                ndarray
                    The image with the contour applied.
                """
                original_image = image.copy()
                # Split the original image into Blue, Green, and Red channels
                b_channel, g_channel, r_channel = cv2.split(original_image)

                
                # Assuming b_channel is the blue channel of your image and contour is your mask
                b_channel[contour == 255] = 0
                g_channel[contour == 255] = 0
                r_channel[contour == 255] = 255

                # Merge the channels back
                return cv2.merge([b_channel, g_channel, r_channel])

        def plot_peak_bounds(self, channel):
            """
            Plots the histogram of a given channel and highlights the main peak bounds.

            This method calculates the histogram of the specified channel, identifies 
            the main peak, and determines its lower and upper bounds. It then plots 
            the histogram and marks the peak bounds with vertical lines.

            Parameters
            ----------
            channel : ndarray
                The channel data (e.g., a color channel from an image) for which the 
                histogram and peak bounds are to be plotted.

            Returns
            -------
            None
            """
            # Calculate the histogram of the channel
            histogram, bin_edges = np.histogram(channel, bins=256, range=(0, 256))

            # Find the bin with the maximum frequency (main peak)
            main_peak_bin = np.argmax(histogram)
            main_peak_value = bin_edges[main_peak_bin]

            # Determine the lower and upper bounds of the main peak
            lower_bound = main_peak_value
            upper_bound = main_peak_value

            # Move left to find the lower bound
            while lower_bound > 0 and histogram[main_peak_bin] == histogram[int(lower_bound) - 1]:
                lower_bound -= 1

            # Move right to find the upper bound
            while upper_bound < 255 and histogram[main_peak_bin] == histogram[int(upper_bound) + 1]:
                upper_bound += 1

            # Plot the histogram with lines representing the bounds
            plt.figure(figsize=(10, 6))
            plt.hist(channel, bins=256, range=(0, 256), color='b', alpha=0.7)
            plt.axvline(x=lower_bound, color='r', linestyle='--', label='Lower Bound')
            plt.axvline(x=upper_bound, color='g', linestyle='--', label='Upper Bound')
            plt.legend()
            plt.title('Histogram with Peak Bounds')
            plt.xlabel('Hue Value')
            plt.ylabel('Frequency')
            plt.show()

            # Print the lower and upper bound values
            print(f"Lower Bound: {lower_bound}")
            print(f"Upper Bound: {upper_bound}")

    class ImageTools:
        """
        A class for performing various image processing operations.

        This class offers functionality to extract specific color channels, resize images,
        and define a region of interest (ROI) on an image. The ROI can be interactively 
        selected by the user using mouse events.

        Parameters
        ----------
        image : ndarray
            The image on which processing operations will be performed.

        Attributes
        ----------
        original_image : ndarray
            A copy of the original image.
        roi : tuple or None
            The coordinates of the defined region of interest (ROI), if defined.

        Methods
        -------
        get_channel(channel_name)
            Extracts and returns a specific channel from the image.
        resize_image(new_width)
            Resizes the image to a new width while maintaining aspect ratio.
        get_image_shape()
            Returns the shape of the image (height, width).
        define_image_roi(channel)
            Interactively defines a region of interest (ROI) on the given channel.
        """
        def __init__(self, image):
            self.image = image
            self.original_image = image.copy()
            self.roi = None

        def get_channel(self, channel_name):
            """
            Extracts and returns a specific color or HSL channel from the image.

            This method allows extraction of individual color channels (Blue, Green, Red) 
            or HSL (Hue, Saturation, Luminance) channels from the image. The channel 
            to be extracted is specified by the channel name.

            Parameters
            ----------
            channel_name : str
                Name of the channel to be extracted. Valid options are "blue", "green", 
                "red", "hue", "saturation", and "luminance".

            Returns
            -------
            ndarray
                The extracted channel as a single-channel image.

            Raises
            ------
            ValueError
                If an invalid channel name is provided.

            Notes
            -----
            - The channel name is not case-sensitive.
            """
            channel_name = channel_name.lower()
            if channel_name in ["blue", "green", "red"]:
                channel_index = {"blue": 0, "green": 1, "red": 2}[channel_name]
                return self.image[:,:,channel_index]
            elif channel_name in ["hue", "saturation", "luminance"]:
                hsl_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)
                channel_index = {"hue": 0, "saturation": 1, "luminance": 2}[channel_name]
                return hsl_image[:,:,channel_index]
            else:
                raise ValueError("Invalid channel name")

        def resize_image(self, new_width):
            """
            Resizes the image to a new width while maintaining the aspect ratio.

            This method resizes the image to the specified width. The height is 
            automatically adjusted to maintain the original aspect ratio of the image.

            Parameters
            ----------
            new_width : int
                The new width to which the image will be resized.

            Returns
            -------
            ndarray
                The resized image.
            """
            height, width = self.image.shape[:2]
            aspect_ratio = height / width
            new_height = int(new_width * aspect_ratio)
            self.image = cv2.resize(self.image, (new_width, new_height))
            return self.image

        def get_image_shape(self):
            """
            Returns the shape (height, width) of the image.

            This method retrieves the dimensions of the image, specifically its height 
            and width, and returns them.

            Returns
            -------
            tuple
                A tuple containing the height and width of the image.
            """
            return self.image.shape[:2]  # Returns (height, width)


        def define_image_roi(self, channel):
            """
            Interactively defines a region of interest (ROI) on the given channel.

            This method allows the user to select a rectangular ROI on the provided
            image channel. The selection is done using mouse events: clicking and 
            dragging to define the rectangle, and releasing to finalize it. The 
            coordinates of the defined ROI are stored.

            Parameters
            ----------
            channel : ndarray
                The single-channel image (e.g., a color or grayscale channel) on which 
                the ROI is to be defined.

            Returns
            -------
            tuple
                A tuple containing the cropped ROI and the coordinates of the ROI 
                in the format (x1, y1, x2, y2).

            Notes
            -----
            - Click and drag to start drawing the rectangle.
            - Release the click to finalize the rectangle.
            - The ROI is defined by the starting and ending coordinates of the rectangle.
            """
            if channel is None:
                print("Error: Could not read the channel.")
                return

            roi = []  # Local variable for the ROI
            drawing = False  # Flag to indicate when the mouse is being dragged

            # Convert the single-channel image to a 3-channel image for drawing
            display_image = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)

            def select_roi(event, x, y, flags, param):
                nonlocal roi, display_image, drawing

                if event == cv2.EVENT_LBUTTONDOWN:
                    roi = [(x, y)]  # Record the starting point on the first click
                    drawing = True

                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        # Update the display image with the rectangle
                        temp_image = display_image.copy()
                        cv2.rectangle(temp_image, roi[0], (x, y), (0, 255, 0), 2)
                        cv2.imshow('Select ROI', temp_image)

                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    roi.append((x, y))  # Record the ending point on the second click

                    # Draw the final rectangle on the display image
                    cv2.rectangle(display_image, roi[0], roi[1], (0, 255, 0), 2)
                    cv2.imshow('Select ROI', display_image)

            # Create a window and set a mouse callback
            cv2.namedWindow('Select ROI')
            cv2.setMouseCallback('Select ROI', select_roi)

            # Show the channel and wait until the user selects the ROI
            while True:
                if not drawing:  # Update the display only when not drawing
                    cv2.imshow('Select ROI', display_image)
                key = cv2.waitKey(1) & 0xFF
                if len(roi) == 2:  # Two points (one rectangle) have been selected
                    break

            cv2.destroyAllWindows()

            # Crop the selected ROI and calculate its histogram
            x1, y1, x2, y2 = roi[0][0], roi[0][1], roi[1][0], roi[1][1]
            roi_cropped = channel[y1:y2, x1:x2]
            # Save the roi
            roi_coordinates = (x1, y1, x2, y2)

            return roi_cropped, roi_coordinates
        
    class MorphologicalManipulation:
        """
        A class for performing morphological operations on images.

        This class allows the application of morphological operations like erosion and 
        dilation on images. A structuring element (kernel) must be set before performing 
        these operations. The class also provides a method to extract mask contours from 
        an image.

        Attributes
        ----------
        image : ndarray
            The original image for manipulation.
        kernel : ndarray
            The structuring element used for morphological operations.
        manipulated_image : ndarray
            The image after applying morphological operations.

        Methods
        -------
        set_kernel(size)
            Sets the structuring element for morphological operations.
        erode(iterations=1)
            Applies erosion to the image.
        dilate(iterations=1)
            Applies dilation to the image.
        get_image()
            Returns the manipulated image.
        get_mask_contour()
            Extracts and returns contours from the image.
        """
        def __init__(self, image):
            self.image = image
            self.kernel = None
            self.manipulated_image = image.copy()

        def set_kernel(self, size):
            """
            Sets the structuring element (kernel) for morphological operations.

            This method initializes the kernel used for erosion and dilation operations. 
            The kernel is a square matrix of the specified size.

            Parameters
            ----------
            size : int
                The size of the structuring element. Represents both the width and height 
                of the square matrix.

            Returns
            -------
            None
            """
            self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

        def erode(self, iterations=1):
            """
            Applies erosion to the manipulated image using the set kernel.

            This method erodes the image, which typically reduces the size of the 
            foreground objects. A kernel must be set before calling this method.

            Parameters
            ----------
            iterations : int, optional
                The number of times the erosion is applied (default is 1).

            Raises
            ------
            ValueError
                If the kernel is not set prior to calling this method.

            Returns
            -------
            None
            """
            if self.kernel is not None:
                self.manipulated_image = cv2.erode(self.manipulated_image, self.kernel, iterations=iterations)
            else:
                raise ValueError("Kernel not set")

        def dilate(self, iterations=1):
            """
            Applies dilation to the manipulated image using the set kernel.

            This method dilates the image, typically increasing the size of foreground 
            objects. A kernel must be set before calling this method.

            Parameters
            ----------
            iterations : int, optional
                The number of times the dilation is applied (default is 1).

            Raises
            ------
            ValueError
                If the kernel is not set prior to calling this method.

            Returns
            -------
            None
            """
            if self.kernel is not None:
                self.manipulated_image = cv2.dilate(self.manipulated_image, self.kernel, iterations=iterations)
            else:
                raise ValueError("Kernel not set")

        def get_image(self):
            """
            Returns the current state of the manipulated image.

            This method retrieves the image as it is after applying morphological 
            operations like erosion or dilation.

            Returns
            -------
            ndarray
                The manipulated image.
            """
            return self.manipulated_image
        
        def get_mask_contour(self):
            """
            Extracts and returns the contours from the manipulated image.

            This method first dilates the image to enhance the contours. It then performs 
            bitwise operations to isolate and return only the contours. The original state 
            of the manipulated image is preserved.

            Returns
            -------
            ndarray
                The image with only the contours extracted.
            """
            # Get the actual mask
            mask = self.get_image()
            mask_dilated = self.get_image()
            # Inverse the mask
            inverse_mask = cv2.bitwise_not(mask)

            # Dilate the mask 
            self.dilate()
            mask_dilated = self.get_image()
            # Perform a bitwise AND between mask1 and the inverse of mask2
            result_mask = cv2.bitwise_and(mask_dilated, inverse_mask)
            # BAckup the old mask before dilatation
            backup_image = self.manipulated_image.copy()
            self.manipulated_image = result_mask
            self.dilate()
            result = self.get_image()
            # Restore the backuped image
            self.manipulated_image = backup_image
            # Keep only closed shape
            return result


    class Thresholding:
        """
        A class for applying various thresholding techniques on image channels.

        This class includes methods for standard and adaptive thresholding, as well as
        utility functions for applying masks to images. It allows for flexible control
        over thresholding parameters and mask application.

        Methods
        -------
        apply_threshold(channel, threshold_type="standard", thresh_value=127, lower_bound=0, upper_bound=255)
            Applies the specified type of thresholding to an image channel.
        get_full_mask(roi_coordinates, original_shape, thresholded_roi)
            Creates and returns a full mask based on the thresholded ROI.
        apply_mask_on_image(original_image, full_mask)
            Applies a mask to the original image and returns the masked image.
        """
        def __init__(self):
            pass

        def apply_threshold(self, channel, threshold_type="standard", thresh_value=127, lower_bound=0, upper_bound=255):
            """
            Applies a specified type of thresholding to an image channel.

            This method supports various thresholding types including standard, Otsu's, 
            and manual. It can work on grayscale or color channels (which are converted 
            to grayscale internally).

            Parameters
            ----------
            channel : ndarray
                The image channel to which the threshold will be applied.
            threshold_type : str, optional
                The type of thresholding to apply. Options are "standard", "otsu", 
                and "manual" (default is "standard").
            thresh_value : int, optional
                The threshold value used in standard thresholding (default is 127).
            lower_bound : int, optional
                The lower bound for manual thresholding (default is 0).
            upper_bound : int, optional
                The upper bound for manual thresholding (default is 255).

            Returns
            -------
            ndarray
                The thresholded image channel.

            Raises
            ------
            ValueError
                If no image channel is loaded or an invalid threshold type is specified.

            Notes
            -----
            - If the image channel is colored, it will be converted to grayscale.
            """
            if channel is None:
                raise ValueError("No image loaded")

            # Convert to grayscale if the image is colored
            if len(channel.shape) == 3:
                channel_image = cv2.cvtColor(channel, cv2.COLOR_BGR2GRAY)
            else:
                channel_image = channel.copy()

            if threshold_type == "otsu":
                _, thresh_image = cv2.threshold(channel_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif threshold_type == "standard":
                _, thresh_image = cv2.threshold(channel_image, thresh_value, 255, cv2.THRESH_BINARY)
            elif threshold_type == "manual":
                thresh_image = cv2.inRange(channel_image, lower_bound, upper_bound)
            else:
                raise ValueError("Invalid threshold type")

            return thresh_image
        
        def get_full_mask(self, roi_coordinates, original_shape, thresholded_roi):
            """
            Creates and returns a full mask from a thresholded region of interest (ROI).

            This method places the thresholded ROI on a full mask the size of the original 
            image, effectively creating a mask that can be applied to the entire image.

            Parameters
            ----------
            roi_coordinates : tuple
                The coordinates of the ROI in the format (x1, y1, x2, y2).
            original_shape : ndarray
                The original shape of the image to match the mask size.
            thresholded_roi : ndarray
                The thresholded region of interest.

            Returns
            -------
            ndarray
                The full mask with the thresholded ROI.
            """
            # Get the shape of the originalk image
            mask_shape = original_shape.shape[:2] 
            # Create a mask with the same dimensions as the original image
            full_mask = np.zeros(mask_shape, dtype=np.uint8)

            # Extract coordinates
            x1, y1, x2, y2 = roi_coordinates

            # Place the thresholded ROI in the full mask
            full_mask[y1:y2, x1:x2] = thresholded_roi

            return full_mask
        
        def apply_mask_on_image(self, original_image, full_mask):
            """
            Applies a binary mask to an image, returning the masked image.

            This method applies a full mask to the original image, masking out areas 
            outside the mask. It ensures the mask is binary before application.

            Parameters
            ----------
            original_image : ndarray
                The original image to which the mask will be applied.
            full_mask : ndarray
                The binary mask to be applied to the original image.

            Returns
            -------
            ndarray
                The original image with the mask applied.

            Raises
            ------
            ValueError
                If either the original image or the mask is not provided.
            """
            if original_image is None or full_mask is None:
                raise ValueError("Image or mask is not provided")

            # Ensure the mask is binary
            _, binary_mask = cv2.threshold(full_mask, 127, 255, cv2.THRESH_BINARY)

            # Apply the mask
            masked_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
            
            return masked_image
        
            
    class Examples:
        """
        A class for demonstrating example use cases of plot and image vizualisation. (Only Demo)

        This class is designed to showcase practical implementations of the techniques and tools
        available in the `LabImageProcessing` package. 

        """
        def __init__(self):
            pass
        def create_sinus_function(self):
            """
            Generates a sinusoidal function.

            This method creates and returns a sine wave function based on a range of 
            values from 0 to 10.

            Returns
            -------
            ndarray
                The sine wave values corresponding to the x values.
            """
            x = np.linspace(0, 10, 100)
            return np.sin(x)
        
        def plot_function(self, function):
            """
            Plots a given mathematical function.

            This method takes a set of y-values corresponding to a function and plots 
            them against a linearly spaced x-axis, creating a visual representation of 
            the function.

            Parameters
            ----------
            function : ndarray
                The y-values of the function to be plotted.

            Returns
            -------
            None
            """
            # Sample data
            x = np.linspace(0, 10, 100)
            y = function

            # Creating the plot
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, '-b', label='Sine Wave')
            plt.title("Simple Plot Example")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.legend(loc='upper right')
            plt.grid(True)

            # Display the plot
            plt.show()

        def show_example_image(self):
            """
            Create and display an example image with OpenCV.

            This function generates an image with a dark blue background and displays white text 
            saying "Example Image". It is primarily designed for demonstration purposes to showcase 
            basic OpenCV functionalities such as creating images, adding text, and handling window 
            events. The function creates a window titled "Press 'Q' to exit", which displays the image. 
            The window remains open until the user presses the 'Q' key.

            The image is created using a numpy array to define its dimensions and color. The text is 
            added to the image using OpenCV's `putText` function. The window handling is managed using 
            OpenCV's `imshow` and `waitKey` functions.
            """

            # Image dimensions and color (dark blue background)
            width, height = 640, 480
            blue_color = (139, 70, 70)  # Dark blue in BGR format

            # Create a blank image with the blue background
            image = np.zeros((height, width, 3), np.uint8)
            image[:] = blue_color

            # Text settings
            text = "Example Image"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)  # White color
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2

            # Put text on the image
            cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, thickness)

            # Display the image in a window
            window_title = "Press 'Q' to exit"
            cv2.imshow(window_title, image)

            # Wait for the 'Q' key to be pressed to exit
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Destroy all OpenCV windows
            cv2.destroyAllWindows()


# Usage Example
# opener = LabImageProcessing.ImageOpener()
# opener.open_image()
# image = opener.get_image()

# morph = LabImageProcessing.MorphologicalManipulation(image)
# morph.set_kernel(5)
# eroded_image = morph.erode(1)

# tools = LabImageProcessing.ImageTools(image)
# blue_channel = tools.get_channel("blue")
# resized_image = tools.resize_image(300)
