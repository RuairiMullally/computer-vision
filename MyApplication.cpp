#include "Utilities.h"
#include <map>

// Test videos and ground truth
char* abandoned_removed_video_files[] = {
	"Video01.avi",
	"Video02.avi",
	"Video03.avi",
	"Video04.avi",
	"Video05.avi",
	"Video06.avi",
	"Video07.avi",
	"Video08.avi",
	"Video09.avi",
	"Video10.avi" };
#define ABANDONED 1
#define REMOVED 2
#define OTHER_CHANGE 3
#define IMAGE_NUMBER_INDEX 0
#define FRAME_NUMBER_INDEX 1
#define CHANGE_TYPE_INDEX 2
#define TOP_ROW_INDEX 3
#define LEFT_COLUMN_INDEX 4
#define BOTTOM_ROW_INDEX 5
#define RIGHT_COLUMN_INDEX 6
// The items in the table below are
//   (1) Video number, (2) Frame number, (3) Type of change, (4) Top row,
//   (5) Left column, (6) Bottom row, (7) Right column
int object_locations[][7] = {
	{1, 115, REMOVED, 105,249,148,311}, // Laptop removed
	{2, 87, ABANDONED, 130,250,148,302}, // Laptop abandoned
	{3, 87, REMOVED, 164,186,223,234}, // Bag removed
	{4, 91, ABANDONED, 208,356,238,388}, // Bag abandoned
	{4, 255, REMOVED, 208,356,238,388}, // Bag removed
	{5, 137, ABANDONED, 118,68,126,76}, // Bag abandoned
	{5, 357, REMOVED, 118,68,126,76}, // Bag removed
	{6, 127, ABANDONED, 28,210,40,227}, // Bag abandoned
	{6, 379, REMOVED, 28,210,40,227}, // Bag removed
	{7, 555, ABANDONED, 108,107,123,127}, // Bag abandoned
	{8, 333, ABANDONED, 104,219,140,275}, // Car abandoned
	{9, 331, ABANDONED, 322,129,376,188}, // Bag abandoned
	{10, 73, OTHER_CHANGE, 109,126,269,225}, // Chair moved
	{10, 83, OTHER_CHANGE, 104,250,148,311} // Laptop opened
};


// Structure to track objects by their component label
struct ObjectTracker
{
	Rect maxBoundingBox;
	int maxArea;
	int stableFrames;
	bool isLocked;
	int lastSeenFrame;
	int classification;
	
	ObjectTracker() : maxArea(0), stableFrames(0), isLocked(false), lastSeenFrame(0), classification(0) {}
};


void classifyAbandonedRemovedObjects(const Mat& currentFrame, const Mat& backgroundFrame, ObjectTracker& tracker, const Mat& persistentMask)
{
    // ---------- ROI + grayscale ----------
    Mat curGray, backGray;
    cvtColor(currentFrame, curGray, COLOR_BGR2GRAY);
    cvtColor(backgroundFrame, backGray, COLOR_BGR2GRAY);
    Rect roi = tracker.maxBoundingBox;
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > curGray.cols || roi.y + roi.height > curGray.rows) return;

    Mat currentROI = curGray(roi).clone();
    Mat backgroundROI = backGray(roi).clone();

    Mat maskROI;
    if (persistentMask.type() == CV_8U)
        maskROI = persistentMask(roi).clone();
    else {
        Mat tmp = persistentMask(roi).clone();
        tmp.convertTo(maskROI, CV_8U, 255.0);
    }
    threshold(maskROI, maskROI, 0, 255, THRESH_BINARY);

    // ---------- Edges → Dilate ----------
    Mat curBlur, backBlur, maskBlur, curEdges, backEdges, maskEdges;
    GaussianBlur(currentROI, curBlur, Size(5,5), 0);
    GaussianBlur(backgroundROI, backBlur, Size(5,5), 0);
    GaussianBlur(maskROI, maskBlur, Size(3,3), 0);

    Canny(curBlur,  curEdges,  50, 150);
    Canny(backBlur, backEdges, 50, 150);
    Canny(maskBlur, maskEdges, 50, 150);

    Mat K = getStructuringElement(MORPH_RECT, Size(2,2));
    Mat curDil, maskDil, backDil;
    dilate(curEdges,  curDil,  K);
    dilate(backEdges, backDil, K);
    dilate(maskEdges, maskDil, K);

    // ---------- Overlap (intersection / mask) ----------
    Mat interCur, interBack;
    bitwise_and(curDil, maskDil, interCur);
    bitwise_and(backDil, maskDil, interBack);

    int maskCount = countNonZero(maskDil);
    if (maskCount == 0) return;
    
    int interCurCount = countNonZero(interCur);
    int interBackCount = countNonZero(interBack);
    
    double overlapCur = double(interCurCount) / double(maskCount);
    double overlapBack = double(interBackCount) / double(maskCount);

    if (overlapCur >= overlapBack)
		tracker.classification = ABANDONED;
	else
		tracker.classification = REMOVED;

    // ---------- Visualization: Include BOTH comparisons ----------
    auto toColor = [](const Mat& g){ Mat c; cvtColor(g, c, COLOR_GRAY2BGR); return c; };

    // Two overlays: one for mask comparison, one for background comparison
    Mat overlayMask = Mat::zeros(curDil.size(), CV_8UC3);
    overlayMask.setTo(Scalar(255,255,255), curDil);      // white for current
    overlayMask.setTo(Scalar(255,0,255),   maskDil);     // magenta for mask
    overlayMask.setTo(Scalar(0,255,0),     interCur); // green for intersection

    Mat overlayBack = Mat::zeros(curDil.size(), CV_8UC3);
    overlayBack.setTo(Scalar(255,255,255), backDil);      // white for current
    overlayBack.setTo(Scalar(0,255,255),   maskDil);     // cyan for background
    overlayBack.setTo(Scalar(0,255,0),     interBack); // green for intersection

    const int TILE_W = 240, TILE_H = 180, GAP = 8;
    const Scalar tileBg(30,30,30);

    auto fitTile = [&](const Mat& src)->Mat {
        Mat srcColor = (src.channels()==1) ? toColor(src) : src;
        double sx = double(TILE_W)/max(1,srcColor.cols);
        double sy = double(TILE_H)/max(1,srcColor.rows);
        double s  = min(sx, sy);
        Mat scaled; resize(srcColor, scaled, Size(round(srcColor.cols*s), round(srcColor.rows*s)), 0,0,
                           (s>=1.0? INTER_NEAREST : INTER_AREA));
        int top = (TILE_H - scaled.rows)/2, bottom = TILE_H - scaled.rows - top;
        int left= (TILE_W - scaled.cols)/2, right  = TILE_W - scaled.cols - left;
        Mat out; copyMakeBorder(scaled, out, top, bottom, left, right, BORDER_CONSTANT, tileBg);
        return out;
    };

    Mat tCur   = fitTile(curDil);
    Mat tMask  = fitTile(maskDil);
    Mat tBack  = fitTile(backDil);
    Mat tOvMask = fitTile(overlayMask);
    Mat tOvBack = fitTile(overlayBack);

    auto label = [&](Mat& img, const string& s){
        rectangle(img, Rect(0,0,img.cols,22), Scalar(0,0,0), FILLED);
        putText(img, s, Point(6,16), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1, LINE_AA);
    };
    label(tCur,     "Current Edges");
    label(tMask,    "Mask Edges");
    label(tBack,    "Background Edges");
    label(tOvMask,  "Current vs Mask (overlap=" + cv::format("%.3f", overlapCur) + ")");
    label(tOvBack,  "Background vs Mask (overlap=" + cv::format("%.3f", overlapBack) + ")");

    // Two rows: first row shows the three edge images, second row shows the two comparisons
    int canvasW = 3*TILE_W + 4*GAP;
    int canvasH = 2*TILE_H + 3*GAP + 40;
    Mat canvas(canvasH, canvasW, CV_8UC3, Scalar(20,20,20));

    auto place = [&](const Mat& tile, int row, int col){
        int x = GAP + col*(TILE_W+GAP);
        int y = GAP + row*(TILE_H+GAP);
        tile.copyTo(canvas(Rect(x,y,TILE_W,TILE_H)));
    };
    
    // First row: edges
    place(tCur,  0, 0);
    place(tMask, 0, 1);
    place(tBack, 0, 2);
    
    // Second row: overlays (centered)
    int offsetX = (TILE_W + GAP) / 2;
    Mat temp1 = canvas(Rect(GAP + offsetX, GAP + TILE_H + GAP, TILE_W, TILE_H));
    tOvMask.copyTo(temp1);
    Mat temp2 = canvas(Rect(GAP + offsetX + TILE_W + GAP, GAP + TILE_H + GAP, TILE_W, TILE_H));
    tOvBack.copyTo(temp2);

    // Decision text at bottom
    string txt = "Mask overlap=" + cv::format("%.3f", overlapCur) + 
                 " vs Back overlap=" + cv::format("%.3f", overlapBack) +
                 (tracker.classification==ABANDONED ? "  -> ABANDONED" : "  -> REMOVED");
    putText(canvas, txt, Point(GAP, canvasH - 12), FONT_HERSHEY_SIMPLEX, 0.6,
            (tracker.classification==ABANDONED ? Scalar(0,220,0) : Scalar(0,0,220)), 2, LINE_AA);

    static bool winit=false;
    if (!winit) { namedWindow("Edge Comparison", WINDOW_NORMAL); winit=true; }
    imshow("Edge Comparison", canvas);
}



void trackBoundingBoxes(
    const Mat& stats,
    const Mat& centroids,
    int numComponents,
    int MIN_AREA,
    int frame_no,
    int STABILITY_FRAMES,
    Mat& current_frame,
    const Mat& background_frame,
    const Mat& persistentMask,
    std::map<int, ObjectTracker>& objectTrackers,
    int& nextObjectID
){

	// need to track which object IDs were seen this frame
	map<int, bool> seenThisFrame;
	
	// process each component, skipping the background component at 0
	for (int i = 1; i < numComponents; i++)
	{
		int area = stats.at<int>(i, CC_STAT_AREA);
		if (area < MIN_AREA)
			continue; // skip small areas
		
		// crete bounding box for this component
		Rect bbox(stats.at<int>(i, CC_STAT_LEFT),
					stats.at<int>(i, CC_STAT_TOP),
					stats.at<int>(i, CC_STAT_WIDTH),
					stats.at<int>(i, CC_STAT_HEIGHT));
		
		// calculate center point as a simple tracking key
		Point2d center = Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
		
		// try to match with existing tracker based on proximity
		int matchedID = -1;
		float minDist = 50.0f; // max distance to consider a match
		
		for (auto& pair : objectTrackers)
		{
			int objID = pair.first;
			ObjectTracker& tracker = pair.second;
			
			// calculate center of existing tracker
			Point2d trackerCenter(tracker.maxBoundingBox.x + tracker.maxBoundingBox.width / 2.0,
									tracker.maxBoundingBox.y + tracker.maxBoundingBox.height / 2.0);
			
			float dist = sqrt(pow(center.x - trackerCenter.x, 2) + pow(center.y - trackerCenter.y, 2));
			
			if (dist < minDist)
			{
				minDist = dist;
				matchedID = objID;
			}
		}
		
		// ff no match found, create new tracker
		if (matchedID == -1)
		{
			matchedID = nextObjectID++;
			objectTrackers[matchedID] = ObjectTracker();
			objectTrackers[matchedID].maxBoundingBox = bbox;
			objectTrackers[matchedID].maxArea = area;
		}
		else
		{
			// update existing tracker if not locked
			ObjectTracker& tracker = objectTrackers[matchedID];
			
			if (!tracker.isLocked)
			{
				if (area > tracker.maxArea)
				{
					// object grew
					tracker.maxBoundingBox = bbox;
					tracker.maxArea = area;
					tracker.stableFrames = 0;
				}
				else
				{
					// object stable
					tracker.stableFrames++;
					if (tracker.stableFrames >= STABILITY_FRAMES) // reached stability threshold to lock an object in place
					{
						tracker.isLocked = true;
						classifyAbandonedRemovedObjects(current_frame, background_frame, tracker, persistentMask);
					}
				}
			}
		}
		
		objectTrackers[matchedID].lastSeenFrame = frame_no;
		seenThisFrame[matchedID] = true;
	}
	
	// draw all tracked objects
	for (auto& pair : objectTrackers)
	{
		int objID = pair.first;
		ObjectTracker& tracker = pair.second;
		
		// Only draw if locked or recently seen
		if (tracker.isLocked || (frame_no - tracker.lastSeenFrame) < 5)
		{
			Scalar color = tracker.isLocked ? Scalar(255, 0, 0) : Scalar(0, 255, 255);
			int thickness = tracker.isLocked ? 2 : 1;
			rectangle(current_frame, tracker.maxBoundingBox, color, thickness);
			
			string label = tracker.isLocked ? 
				("Locked " + to_string(objID)) : 
				("Track " + to_string(objID));
			if (tracker.isLocked && tracker.classification != 0)
			{
				string classLabel = (tracker.classification == ABANDONED) ? " ABANDONED" :
									(tracker.classification == REMOVED) ? " REMOVED" : " OTHER";
				label += classLabel;
			}
			putText(current_frame, label, 
				Point(tracker.maxBoundingBox.x, tracker.maxBoundingBox.y - 5),
				FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
		}
	}
}


void MyApplication()
{
	char* file_location = "Media/Abandoned/";
	int number_of_videos = sizeof(abandoned_removed_video_files) / sizeof(abandoned_removed_video_files[0]);
	VideoCapture* video = new VideoCapture[number_of_videos];
	for (int video_file_no = 1; (video_file_no <= number_of_videos); video_file_no++)
	{
		string filename(file_location);
		filename.append(abandoned_removed_video_files[video_file_no-1]);
		video[video_file_no-1].open(filename);
		if (video[video_file_no-1].isOpened())
		// At this point the video file is open and we can process it
		{
			int frame_no = 1;
			Mat current_frame;
			// store the first frame in current_frame
			video[video_file_no-1] >> current_frame;
			Mat background_frame = current_frame.clone(); //bavkground frame used for classification

			//GMM Background Subtractors
			cv::Ptr<cv::BackgroundSubtractorMOG2> fastGMM = cv::createBackgroundSubtractorMOG2();
			fastGMM->setVarThreshold(75);
			fastGMM->setDetectShadows(true);
			fastGMM->setHistory(350);
			fastGMM->setShadowThreshold(0.5);

			cv::Ptr<cv::BackgroundSubtractorMOG2> slowGMM = cv::createBackgroundSubtractorMOG2();
			slowGMM->setVarThreshold(75);
			slowGMM->setDetectShadows(true);
			slowGMM->setHistory(350);
			slowGMM->setShadowThreshold(0.5);

			//Kernels:
			Mat kOpen = getStructuringElement(MORPH_RECT, Size(3, 3));
			Mat kClose = getStructuringElement(MORPH_RECT, Size(7, 7));

			const int PERSIST_FRAMES = 50; // number of frames a change must persist to be added to the persistent mask
			const int MIN_AREA = 25; // minimum area of a region to be considered an object
			const int STABILITY_FRAMES = 25;  // frames to wait before locking
			Mat persistCount(current_frame.rows, current_frame.cols, CV_16U, Scalar(0)); // counting matrix to track persistence
			Mat persistentMask(current_frame.rows, current_frame.cols, CV_16U, Scalar(0)); // once a pixel reaches threshold in persistCount, it is added here
			
			// Map to track objects by their spatial location (center point as key)
			map<int, ObjectTracker> objectTrackers;
			int nextObjectID = 1;

			// now loop through the video, comparing each frame with the background frame
			while (!current_frame.empty())
			{
				//compute the foreground masks
				Mat difference_frame, difference_frame2;
				fastGMM->apply(current_frame, difference_frame, 0.0015);
				slowGMM->apply(current_frame, difference_frame2, 0.0003);
				Mat fastfgBinary = (difference_frame == 255); // to remove shadow pixels
				Mat slowfgBinary = (difference_frame2 == 255);

				morphologyEx(fastfgBinary, fastfgBinary, MORPH_OPEN, kOpen); //open and close to join objects
				morphologyEx(fastfgBinary, fastfgBinary, MORPH_CLOSE, kClose);
				morphologyEx(slowfgBinary, slowfgBinary, MORPH_OPEN, kOpen);
				morphologyEx(slowfgBinary, slowfgBinary, MORPH_CLOSE, kClose);
				
				Mat changeCandidate; // something that is in the slow mask and not the fast mask is a potential object
				bitwise_and(slowfgBinary, ~fastfgBinary, changeCandidate);
				morphologyEx(changeCandidate, changeCandidate, MORPH_OPEN, kOpen);
           		morphologyEx(changeCandidate, changeCandidate, MORPH_CLOSE, kClose);

				persistCount.setTo(0, changeCandidate == 0);  // reset where background
				add(persistCount, Scalar(1), persistCount, changeCandidate); // increment where remains changed
				compare(persistCount, Scalar(PERSIST_FRAMES), persistentMask, CMP_GE); // update persistent mask if the change has been persistent enough
				
				// sse connected components to label each separate region
				Mat labels, stats, centroids;
				persistentMask.convertTo(persistentMask, CV_8U);
				int numComponents = connectedComponentsWithStats(persistentMask, labels, stats, centroids);


				//use the GMM slow background for classification
				Mat background_frame;
				slowGMM->getBackgroundImage(background_frame);
				trackBoundingBoxes(
					stats,
					centroids,
					numComponents,
					MIN_AREA,
					frame_no,
					STABILITY_FRAMES,
					current_frame,
					background_frame,
					persistentMask,
					objectTrackers,
					nextObjectID
				);

				// Draw ground truth
				for (int current = 0; (current < sizeof(object_locations) / 7); current++)
				{
					if ((object_locations[current][IMAGE_NUMBER_INDEX] == video_file_no) && (object_locations[current][FRAME_NUMBER_INDEX] <= frame_no))
					{
						Scalar colour((object_locations[current][CHANGE_TYPE_INDEX] == OTHER_CHANGE) ? 0xFF : 0x00,
							(object_locations[current][CHANGE_TYPE_INDEX] == ABANDONED) ? 0xFF : 0x00,
							(object_locations[current][CHANGE_TYPE_INDEX] == REMOVED) ? 0xFF : 0x00 );
						rectangle(current_frame, Point(object_locations[current][LEFT_COLUMN_INDEX], object_locations[current][TOP_ROW_INDEX]),
							Point(object_locations[current][RIGHT_COLUMN_INDEX], object_locations[current][BOTTOM_ROW_INDEX]), colour, 1, 8, 0);
					}
				}
				// Display the original frame
				string frame_title("Frame no ");
				frame_title.append(std::to_string(frame_no));
				writeText(current_frame, frame_title, 15, 15);
				imshow(string("Original - ") + abandoned_removed_video_files[video_file_no-1], current_frame);

				// Display the difference frame
				writeText(fastfgBinary, frame_title, 15, 15);
				imshow(string("Difference - ") + abandoned_removed_video_files[video_file_no-1], fastfgBinary);

				writeText(slowfgBinary, frame_title, 15, 15);
				imshow(string("Difference2 - ") + abandoned_removed_video_files[video_file_no-1], slowfgBinary);

				writeText(persistentMask, frame_title, 15, 15);
				imshow(string("persistentMask - ") + abandoned_removed_video_files[video_file_no-1], persistentMask);

				char choice = cv::waitKey(25);  // Delay between frames
				if (choice == 'w')
				{
					break;
				}
				video[video_file_no-1] >> current_frame;
				frame_no++;
			}
			while (true)
			{
				char key = cv::waitKey(0);  // Wait indefinitely for key press
				if (key == 'w')
				{
					break;
				}
			}
			destroyWindow(string("Original - ") + abandoned_removed_video_files[video_file_no-1]);
			destroyWindow(string("Difference - ") + abandoned_removed_video_files[video_file_no-1]);
			destroyWindow(string("Difference2 - ") + abandoned_removed_video_files[video_file_no-1]);
			destroyWindow(string("persistentMask - ") + abandoned_removed_video_files[video_file_no-1]);
			
		}
		else
		{
			cout << "Cannot open video file: " << filename << endl;
			//			return -1;
		}
	}

}