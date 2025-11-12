#include "Utilities.h"
#include <map>
#include <vector>
#include <iomanip>

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

// Structure to hold detection results for scoring
struct Detection {
	int videoNum;
	int frameNum;
	int classification;
	Rect boundingBox;
	int objectID;
};

// Structure to hold aggregate scoring statistics
struct AggregateStats {
	int totalGroundTruth;
	int totalTruePositives;
	int totalFalsePositives;
	int totalFalseNegatives;
	int totalCorrectClassifications;
	double totalDelay;
	double totalOverlap;
	int numDelayMeasurements;
	int numOverlapMeasurements;
	
	AggregateStats() : totalGroundTruth(0), totalTruePositives(0), totalFalsePositives(0),
		totalFalseNegatives(0), totalCorrectClassifications(0), totalDelay(0.0), 
		totalOverlap(0.0), numDelayMeasurements(0), numOverlapMeasurements(0) {}
};
AggregateStats globalStats;

// intersection over union for 2 bounding boxes
double calculateIoU(const Rect& box1, const Rect& box2) {
	int x1 = max(box1.x, box2.x);
	int y1 = max(box1.y, box2.y);
	int x2 = min(box1.x + box1.width, box2.x + box2.width);
	int y2 = min(box1.y + box1.height, box2.y + box2.height);
	
	if (x2 < x1 || y2 < y1)
		return 0.0; //no overlap case
	
	int intersectionArea = (x2 - x1) * (y2 - y1);
	int box1Area = box1.width * box1.height;
	int box2Area = box2.width * box2.height;
	int unionArea = box1Area + box2Area - intersectionArea;
	
	return double(intersectionArea) / double(unionArea);
}

void scoreAlgorithm(const vector<Detection>& detections, int videoNum) {
	const double IOU_THRESHOLD = 0.3; // iou matching thres
	
	cout << "\n========== Video " << videoNum << " - Comparison ==========" << endl;
	
	// print ground truth for this video
	cout << "\nGROUND TRUTH:" << endl;
	vector<int> gtIndices;
	for (int i = 0; i < sizeof(object_locations) / sizeof(object_locations[0]); i++) {
		if (object_locations[i][IMAGE_NUMBER_INDEX] == videoNum) {
			gtIndices.push_back(i); // store GT for this video

			string classType = (object_locations[i][CHANGE_TYPE_INDEX] == ABANDONED) ? "ABANDONED" : 
                   ((object_locations[i][CHANGE_TYPE_INDEX] == REMOVED) ? "REMOVED" : "OTHER_CHANGE");

			cout << "  GT" << gtIndices.size() << ": Frame=" << object_locations[i][FRAME_NUMBER_INDEX]
				 << ", Type=" << classType
				 << ", Box=[" << object_locations[i][LEFT_COLUMN_INDEX] << "," 
				 << object_locations[i][TOP_ROW_INDEX] << "," 
				 << object_locations[i][RIGHT_COLUMN_INDEX] << ","
				 << object_locations[i][BOTTOM_ROW_INDEX] << "]" << endl;
		}
	}
	
	int numGroundTruth = gtIndices.size();
	
	// printing detections for this video
	cout << "DETECTED OBJECTS:" << endl;
	vector<Detection> videoDetections;
	for (const auto& det : detections) {
		if (det.videoNum == videoNum) {
			videoDetections.push_back(det);

			string classType = (det.classification == ABANDONED) ? "ABANDONED" : "REMOVED";
			cout << "  DET" << videoDetections.size() << ": Frame=" << det.frameNum
				 << ", Type=" << classType
				 << ", Box=[" << det.boundingBox.x << "," << det.boundingBox.y << ","
				 << (det.boundingBox.x + det.boundingBox.width) << ","
				 << (det.boundingBox.y + det.boundingBox.height) << "]" << endl;
		}
	}
	
	// matching detections to ground truth
	cout << "MATCHING:" << endl;
	int truePositives = 0;
	int falsePositives = 0;
	int correctClassifications = 0;
	vector<bool> groundTruthMatched(sizeof(object_locations) / sizeof(object_locations[0]), false);
	
	for (size_t detIdx = 0; detIdx < videoDetections.size(); detIdx++) {
		const Detection& det = videoDetections[detIdx];
		
		bool matched = false;
		int bestMatch = -1;
		double bestIoU = 0.0;
		
		// finding best matching ground truth
		for (size_t gtIdx = 0; gtIdx < gtIndices.size(); gtIdx++) {
			int i = gtIndices[gtIdx];
			
			if (groundTruthMatched[i])
				continue; // already matched
			
			Rect gtBox(object_locations[i][LEFT_COLUMN_INDEX],
					   object_locations[i][TOP_ROW_INDEX],
					   object_locations[i][RIGHT_COLUMN_INDEX] - object_locations[i][LEFT_COLUMN_INDEX],
					   object_locations[i][BOTTOM_ROW_INDEX] - object_locations[i][TOP_ROW_INDEX]);
			
			double iou = calculateIoU(det.boundingBox, gtBox);
			
			if (iou > bestIoU && iou >= IOU_THRESHOLD) {
				bestIoU = iou;
				bestMatch = i;
				matched = true;
			}
		}
		
		if (matched) {
			truePositives++;
			groundTruthMatched[bestMatch] = true;
			
			// find which GT index this is
			int gtNum = 0;
			for (size_t k = 0; k < gtIndices.size(); k++) {
				if (gtIndices[k] == bestMatch) {
					gtNum = k + 1;
					break;
				}
			}
			
			// calculate delay
			int delay = det.frameNum - object_locations[bestMatch][FRAME_NUMBER_INDEX];
			globalStats.totalDelay += delay;
			globalStats.numDelayMeasurements++;
			
			// store overlap
			globalStats.totalOverlap += bestIoU;
			globalStats.numOverlapMeasurements++;
			
			// check if classification is correct
			int gtClass = object_locations[bestMatch][CHANGE_TYPE_INDEX];
			bool classCorrect = (det.classification == gtClass);
			if (classCorrect) {
				correctClassifications++;
			}
			
			cout << "  DET" << (detIdx + 1) << " -> GT" << gtNum 
				 << " (IoU=" << fixed << setprecision(3) << bestIoU 
				 << ", Delay=" << delay << " frames"
				 << ", Class=" << (classCorrect ? "CORRECT" : "WRONG") << ")" << endl;
		} else {
			falsePositives++;
			cout << "  DET" << (detIdx + 1) << " -> NO MATCH (False Positive)" << endl;
		}
	}
	
	// check for unmatched ground truth
	for (size_t gtIdx = 0; gtIdx < gtIndices.size(); gtIdx++) {
		int i = gtIndices[gtIdx];
		if (!groundTruthMatched[i]) {
			cout << "  GT" << (gtIdx + 1) << " -> NOT DETECTED (False Negative)" << endl;
		}
	}
	
	int falseNegatives = numGroundTruth - truePositives;
	
	//update global stats
	globalStats.totalGroundTruth += numGroundTruth;
	globalStats.totalTruePositives += truePositives;
	globalStats.totalFalsePositives += falsePositives;
	globalStats.totalFalseNegatives += falseNegatives;
	globalStats.totalCorrectClassifications += correctClassifications;
	
	cout << "\nSUMMARY: TP=" << truePositives 
		 << ", FP=" << falsePositives 
		 << ", FN=" << falseNegatives 
		 << ", Correct Class=" << correctClassifications << "/" << truePositives << endl;
	cout << "================================================\n" << endl;
}

void printAggregateResults() {
	cout << "\n\n" << endl;
	cout << "###################################################" << endl;
	cout << "##        AGGREGATE RESULTS - ALL VIDEOS         ##" << endl;
	cout << "###################################################" << endl;
	cout << "\nDETECTION PERFORMANCE:" << endl;
	cout << "  Total Ground Truth Objects: " << globalStats.totalGroundTruth << endl;
	cout << "  True Positives:  " << globalStats.totalTruePositives << endl;
	cout << "  False Positives: " << globalStats.totalFalsePositives << endl;
	cout << "  False Negatives: " << globalStats.totalFalseNegatives << endl;
	
	double precision = (globalStats.totalTruePositives + globalStats.totalFalsePositives > 0) ? 
		double(globalStats.totalTruePositives) / double(globalStats.totalTruePositives + globalStats.totalFalsePositives) : 0.0;
	double recall = (globalStats.totalGroundTruth > 0) ? 
		double(globalStats.totalTruePositives) / double(globalStats.totalGroundTruth) : 0.0;
	double f1Score = (precision + recall > 0) ? 
		2.0 * (precision * recall) / (precision + recall) : 0.0;
	
	cout << "\nOVERALL METRICS:" << endl;
	cout << "  Precision: " << fixed << setprecision(3) << precision << endl;
	cout << "  Recall:    " << fixed << setprecision(3) << recall << endl;
	cout << "  F1-Score:  " << fixed << setprecision(3) << f1Score << endl;
	
	double classificationAccuracy = (globalStats.totalTruePositives > 0) ? 
		double(globalStats.totalCorrectClassifications) / double(globalStats.totalTruePositives) : 0.0;
	cout << "  Classification Accuracy: " << fixed << setprecision(3) << classificationAccuracy 
		 << " (" << globalStats.totalCorrectClassifications << "/" << globalStats.totalTruePositives << " correct)" << endl;
	
	double avgDelay = (globalStats.numDelayMeasurements > 0) ? 
		globalStats.totalDelay / double(globalStats.numDelayMeasurements) : 0.0;
	cout << "  Average Detection Delay: " << fixed << setprecision(2) << avgDelay << " frames" << endl;
	
	double avgOverlap = (globalStats.numOverlapMeasurements > 0) ? 
		globalStats.totalOverlap / double(globalStats.numOverlapMeasurements) : 0.0;
	cout << "  Average Spatial Overlap (IoU): " << fixed << setprecision(3) << avgOverlap << endl;
	
	cout << "\n###################################################\n" << endl;
}


void classifyAbandonedRemovedObjects(const Mat& currentFrame, const Mat& backgroundFrame, ObjectTracker& tracker, const Mat& persistentMask)
{
    // get sub-region and convert to grayscale
    Mat curGray, backGray;
    cvtColor(currentFrame, curGray, COLOR_BGR2GRAY);
    cvtColor(backgroundFrame, backGray, COLOR_BGR2GRAY);
    Rect roi = tracker.maxBoundingBox;
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > curGray.cols || roi.y + roi.height > curGray.rows) return;

    Mat currentROI = curGray(roi).clone();
    Mat backgroundROI = backGray(roi).clone();

	//mask originally defined as CV_16U
    Mat maskROI;
    if (persistentMask.type() == CV_8U)
        maskROI = persistentMask(roi).clone();
    else {
        Mat tmp = persistentMask(roi).clone();
        tmp.convertTo(maskROI, CV_8U, 255.0);
    }
    threshold(maskROI, maskROI, 0, 255, THRESH_BINARY);

    // find edges and dilate
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

    // find overlap
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

	// imshow("Classify ROI", maskDil);
	// imshow("Current Edges", curDil);
	// imshow("Background Edges", backDil);
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
			
			// Vector to store locked detections for scoring
			vector<Detection> lockedDetections;

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
				
				// store locked detections for scoring
				for (auto& pair : objectTrackers) {
					int objID = pair.first;
					ObjectTracker& tracker = pair.second;
					
					if (tracker.isLocked && tracker.classification != 0) {
						// check if already in vector
						bool alreadyAdded = false;
						for (const auto& det : lockedDetections) {
							if (det.objectID == objID) {
								alreadyAdded = true;
								break;
							}
						}
						
						// only add once per locked object
						if (!alreadyAdded) {
							Detection det;
							det.videoNum = video_file_no;
							det.frameNum = frame_no;
							det.classification = tracker.classification;
							det.boundingBox = tracker.maxBoundingBox;
							det.objectID = objID;
							lockedDetections.push_back(det);
						}
					}
				}

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

				char choice = cv::waitKey(5);  // Delay between frames
				if (choice == 'w')
				{
					break;
				}
				video[video_file_no-1] >> current_frame;
				frame_no++;
			}

			scoreAlgorithm(lockedDetections, video_file_no);
			
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

	printAggregateResults();
}