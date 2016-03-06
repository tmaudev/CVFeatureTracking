#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>

#define MIN_FEATURES 30
#define FAST_THRESHOLD 100
#define FEATURE_POINT_SIZE 3

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
   /* Video Camera Object */
   VideoCapture cam;

   /* Video Frame Images */
   Mat frame, frame_gray, oldframe;

   /* Feature Point Vectors */
   std::vector<KeyPoint> keypoints;
   std::vector<Point2f> pts, tracked_pts, prev_pts;

   /* Status and Error Flags for KLT Algorithm */
   std::vector<uchar> status;
   std::vector<float> err;

   /* Number of Current Feature Points */
   int num_pts = 0;

   /* Pixed Distance Calculation Variables */
   float sum_x, sum_y, dist_x = 0, dist_y = 0;
   //int min_x, max_x, min_y, max_y;

   /* Open Camera Device */
   cam.open(0);

   /* Check Camera Device Opens (Returns -1 on Error) */
   if (!cam.isOpened()) {
      return -1;
   }

   /* Create Window for Displaying Video Feed */
   namedWindow("Feature Tracking", WINDOW_NORMAL);

   /* Loop Forever Until Camera Closed */
   while (cam.isOpened()) {
      /* Obtain Frame from Camera */
      cam >> frame;

      /* Convert Frame to Grayscale */
      cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

      /* Check for Minimum Number of Features */
      if (tracked_pts.size() < MIN_FEATURES) {
         /* Find New Features (FAST Algorithm) */
         FAST(frame_gray, keypoints, FAST_THRESHOLD, true);
         KeyPoint::convert(keypoints, pts);

         //goodFeaturesToTrack(frame_gray, pts, 50, 0.01, 0.1);

         /* Keep Track of Feature Points */
         tracked_pts = pts;

         /* Number of Feature Points */
         num_pts = pts.size();
         printf("FAST: Found %d Features\n", num_pts);
      }
      else {
         /* Check if First Frame */
         if (!oldframe.empty()) {

            /* Check Minimum Number of Features */
            if (num_pts >= MIN_FEATURES) {

               /* Track Points (KLT Algorithm) */
               calcOpticalFlowPyrLK(oldframe, frame_gray, tracked_pts, pts, status, err);

               /* Failed Point Tracking */
               if (countNonZero(status) < status.size() * 0.8) {
                  printf("ERROR\n");
                  tracked_pts.clear();
                  oldframe.release();
                  continue;
               }

               /* Save Old Points */
               prev_pts = tracked_pts;

               /* Clear Tracked Points */
               tracked_pts.clear();

               /* Reset Sum Variables */
               sum_x = 0;
               sum_y = 0;

               /* Rebuild Tracked Points Without Failed Points */
               for (int i = 0; i < status.size(); i++) {
                  if(status[i]) {
                     tracked_pts.push_back(pts[i]);

                     /* Calculate Sum of Movement */
                     sum_x += pts[i].x - prev_pts[i].x;
                     sum_y += pts[i].y - prev_pts[i].y;
                  }
               }

               /* Calculate Average Movement in X and Y Directions */
               dist_x += sum_x / tracked_pts.size();
               dist_y += sum_y / tracked_pts.size();
               printf("X: %8.3f        Y: %8.3f\n", dist_x, dist_y);
            }
         }
      }

      /* Plot Feature Points */
      for (int i = 0; i < tracked_pts.size(); i++) {
         circle(frame, tracked_pts[i], FEATURE_POINT_SIZE, Scalar(0, 0, 255), CV_FILLED);
      }

      /* Save Old Frame */
      oldframe = frame_gray;

      /* Release Grayscale Frame */
      frame_gray.release();
    
      /* Display Frame in Window */
      imshow("Live View", frame);

      /* If Return Key Pressed, Exit */
      if (waitKey(1) >= 0) {
         break;
      }
   }

   return 0;
}