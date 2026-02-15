// Images were preprocessed using background correction, CLAHE contrast enhancement, vesselness filtering, adaptive thresholding, and morphological cleanup to generate binary vessel masks for vascular network analysis.
// Define the parent folder
parentDir = "ang_analysis\\";

setBatchMode(true);  // Speeds up processing

indices = newArray(
    1,2,3,4,5...
);

for (j = 0; j < indices.length; j++) {
    i = indices[j];
    // subfolder = parentDir + i + "\\";
    filename = "im_plot_" + i + ".png";
    fullPath = parentDir + filename;

    if (File.exists(fullPath)) {
        print("Processing: " + fullPath);
        open(fullPath);

        originalTitle = getTitle();

        run("8-bit");
        run("Subtract Background...", "rolling=30");
        run("Enhance Local Contrast (CLAHE)", "blocksize=16 histogram=256 maximum=2 mask=*None* fast");
        run("Enhance Contrast", "saturated=0.35");
        run("RGB Color");

        run("Duplicate...", "title=blurred");
        selectWindow("blurred");
        run("Gaussian Blur...", "sigma=50");

        selectWindow(originalTitle);
        imageCalculator("Subtract create 32-bit", originalTitle, "blurred");
        rename("subtracted");

        selectWindow("subtracted");
        run("8-bit");
        run("Tubeness", "sigma=3 use");
        run("8-bit");
        run("Auto Local Threshold", "method=Phansalkar radius=15 parameter_1=0 parameter_2=0 white");
        run("Open");
        run("Erode");
        run("Dilate");
        run("RGB Color");

        // Save result to same folder
        saveAs("PNG", parentDir + "processed_" + i + ".png");

        close("*");  // Close all windows
    } else {
        print("File not found: " + fullPath);
    }
}

setBatchMode(false);
