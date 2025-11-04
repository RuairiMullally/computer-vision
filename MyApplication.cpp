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
	
	ObjectTracker() : maxArea(0), stableFrames(0), isLocked(false), lastSeenFrame(0) {}
};

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
			//store frame 3 of the video as the starting background frame
			Mat background_frame = current_frame.clone();
			// after: Mat background_frame = current_frame.clone();

			//GMM Background Subtractors
			cv::Ptr<cv::BackgroundSubtractorMOG2> mog1 = cv::createBackgroundSubtractorMOG2();
			mog1->setVarThreshold(75);
			mog1->setDetectShadows(true);
			mog1->setHistory(350);
			mog1->setShadowThreshold(0.5);

			cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = cv::createBackgroundSubtractorMOG2();
			mog2->setVarThreshold(75);
			mog2->setDetectShadows(true);
			mog2->setHistory(350);
			mog2->setShadowThreshold(0.5);

			//Kernels:
			Mat kOpen = getStructuringElement(MORPH_RECT, Size(3, 3));
			Mat kClose = getStructuringElement(MORPH_RECT, Size(8, 8));

			const int PERSIST_FRAMES = 50; // number of frames a change must persist to be added to the persistent mask
			const int MIN_AREA = 25; // minimum area of a region to be considered an object
			const int STABILITY_FRAMES = 25;  // frames to wait before locking
			Mat persistCount(current_frame.rows, current_frame.cols, CV_16U, Scalar(0)); // counting matrix to track persistence
			Mat persistentMask(current_frame.rows, current_frame.cols, CV_16U, Scalar(0)); // once a pixel reaches threshold in persistCount, it is added here

			// store labeled regions from previous frame for tracking
			Mat previousLabels;
			
			// Map to track objects by their spatial location (center point as key)
			map<int, ObjectTracker> objectTrackers;
			int nextObjectID = 1;

			// now loop through the video, comparing each frame with the background frame
			while (!current_frame.empty())
			{
				//compute the foreground masks
				Mat difference_frame, difference_frame2;
				mog1->apply(current_frame, difference_frame, 0.0015);
				mog2->apply(current_frame, difference_frame2, 0.0003);
				Mat fgBinary = (difference_frame == 255); // to remove shadow pixels
				Mat fgBinary2 = (difference_frame2 == 255);

				morphologyEx(fgBinary, fgBinary, MORPH_OPEN, kOpen); //open and close to join objects
				morphologyEx(fgBinary, fgBinary, MORPH_CLOSE, kClose);
				morphologyEx(fgBinary2, fgBinary2, MORPH_OPEN, kOpen);
				morphologyEx(fgBinary2, fgBinary2, MORPH_CLOSE, kClose);
				
				Mat changeCandidate;
				bitwise_and(fgBinary2, ~fgBinary, changeCandidate);
				morphologyEx(changeCandidate, changeCandidate, MORPH_OPEN, kOpen);
           		morphologyEx(changeCandidate, changeCandidate, MORPH_CLOSE, kClose);

				persistCount.setTo(0, changeCandidate == 0);  // reset where background
				add(persistCount, Scalar(1), persistCount, changeCandidate); // increment where remains changed
				compare(persistCount, Scalar(PERSIST_FRAMES), persistentMask, CMP_GE); // update persistent mask if the change has been persistent enough
				
				// sse connected components to label each separate region
				Mat labels, stats, centroids;
				persistentMask.convertTo(persistentMask, CV_8U);
				int numComponents = connectedComponentsWithStats(persistentMask, labels, stats, centroids);
				
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
					int centerKey = (int)(center.x / 10) * 10000 + (int)(center.y / 10); // grid-based key
					
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
						putText(current_frame, label, 
							Point(tracker.maxBoundingBox.x, tracker.maxBoundingBox.y - 5),
							FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
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
				writeText(fgBinary, frame_title, 15, 15);
				imshow(string("Difference - ") + abandoned_removed_video_files[video_file_no-1], fgBinary);

				writeText(fgBinary2, frame_title, 15, 15);
				imshow(string("Difference2 - ") + abandoned_removed_video_files[video_file_no-1], fgBinary2);

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