
#include "Utilities.h"


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
			cv::Ptr<cv::BackgroundSubtractorMOG2> mog1 = cv::createBackgroundSubtractorMOG2();
			mog1->setVarThreshold(25);
			mog1->setDetectShadows(true);

			cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = cv::createBackgroundSubtractorMOG2();
			mog2->setVarThreshold(25);
			mog2->setDetectShadows(true);

			const int PERSIST_FRAMES = 30;
			Mat persistCount(current_frame.rows, current_frame.cols, CV_16U, Scalar(0)); 
			Mat persistentMask(current_frame.rows, current_frame.cols, CV_16U, Scalar(0));
			// now loop through the video, comparing each frame with the background frame
			while (!current_frame.empty())
			{
				//compute the absolute difference between the current frame and the background frame
				Mat difference_frame, difference_frame2;
				mog1->apply(current_frame, difference_frame, 0.004);
				mog2->apply(current_frame, difference_frame2, 0.001);
				Mat fgBinary = (difference_frame == 255);
				Mat fgBinary2 = (difference_frame2 == 255);

				morphologyEx(fgBinary, fgBinary, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
				morphologyEx(fgBinary, fgBinary, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(12, 12)));
				morphologyEx(fgBinary2, fgBinary2, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
				morphologyEx(fgBinary2, fgBinary2, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(12, 12)));
				// TODO: if a pixel has been white in the difference frame for 20 consecutive frames, draw a red mask in that position over the original image
				// find a persistent change by considering pixels where slow mask shows change but quick mask does not
				Mat changeCandidate;
				bitwise_and(fgBinary2, ~fgBinary, changeCandidate);
				morphologyEx(fgBinary, fgBinary, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(5, 5))); // apply closing to a candidate

				persistCount.setTo(0, changeCandidate == 0);  // reset where background
				add(persistCount, Scalar(1), persistCount, changeCandidate); // increment where changed
				compare(persistCount, Scalar(PERSIST_FRAMES), persistentMask, CMP_GE);



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

				char choice = cv::waitKey(50);  // Delay between frames
				video[video_file_no-1] >> current_frame;
				frame_no++;
			}

		}
		else
		{
			cout << "Cannot open video file: " << filename << endl;
			//			return -1;
		}
	}

}
