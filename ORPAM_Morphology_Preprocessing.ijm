// This macro enhances vascular contrast and removes uneven background illumination so vessel segmentation and quantitative morphology measurements become reliable.
inputDir = "diam_analysis\\";
setBatchMode(true);

// Define specific indices to process
indices = newArray(
    1,2,3,4,5...
);

for (j = 0; j < indices.length; j++) {
    i = indices[j];
    subDir = inputDir + i + "\\";
    imageName = "im_plot_" + i + ".png";
    fullPath = inputDir + imageName;

    if (File.exists(fullPath)) {
        print("Processing: " + fullPath);

        open(fullPath);
        originalTitle = getTitle();

        run("8-bit");
        run("Enhance Local Contrast (CLAHE)", "blocksize=16 histogram=256 maximum=2 mask=*None* fast");
        run("Enhance Contrast", "saturated=0.35");
        run("RGB Color");

        run("Duplicate...", "title=blurred");
        selectWindow("blurred");
        run("Gaussian Blur...", "sigma=50");

        selectWindow(originalTitle);
        imageCalculator("Subtract create 32-bit", originalTitle, "blurred");
        rename("subtracted");

        saveAs("PNG", inputDir + "im_plot_" + i + "_subtracted.png");

        close("*");
    } else {
        print("File not found: " + fullPath);
    }
}

setBatchMode(false);
