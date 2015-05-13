package java_cv_ImageDifference;

import java.awt.AWTException;
import java.awt.Graphics;
import java.awt.Robot;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

class Panel extends JPanel{  
	  private static final long serialVersionUID = 1L;  
	  private BufferedImage image;    
	  // Create a constructor method  
	  public Panel(){  
	    super();  
	  }  
	  private BufferedImage getimage(){  
	    return image;  
	  }  
	  public void setimage(BufferedImage newimage){  
	    image=newimage;  
	    return;  
	  }  
	  public void setimagewithMat(Mat newimage){  
	    image=this.matToBufferedImage(newimage);  
	    return;  
	  }  
	  /**  
	   * Converts/writes a Mat into a BufferedImage.  
	   *  
	   * @param matrix Mat of type CV_8UC3 or CV_8UC1  
	   * @return BufferedImage of type TYPE_3BYTE_BGR or TYPE_BYTE_GRAY  
	   */  
	  public BufferedImage matToBufferedImage(Mat matrix) {  
	    int cols = matrix.cols();  
	    int rows = matrix.rows();  
	    int elemSize = (int)matrix.elemSize();  
	    byte[] data = new byte[cols * rows * elemSize];  
	    int type;  
	    matrix.get(0, 0, data);  
	    switch (matrix.channels()) {  
	      case 1:  
	        type = BufferedImage.TYPE_BYTE_GRAY;  
	        break;  
	      case 3:  
	        type = BufferedImage.TYPE_3BYTE_BGR;  
	        // bgr to rgb  
	        byte b;  
	        for(int i=0; i<data.length; i=i+3) {  
	          b = data[i];  
	          data[i] = data[i+2];  
	          data[i+2] = b;  
	        }  
	        break;  
	      default:  
	        return null;  
	    }  
	    BufferedImage image2 = new BufferedImage(cols, rows, type);  
	    image2.getRaster().setDataElements(0, 0, cols, rows, data);  
	    return image2;  
	  }  
	  @Override  
	  protected void paintComponent(Graphics g){  
	     super.paintComponent(g);  
	     //BufferedImage temp=new BufferedImage(640, 480, BufferedImage.TYPE_3BYTE_BGR);  
	     BufferedImage temp=getimage();  
	     //Graphics2D g2 = (Graphics2D)g;
	     if( temp != null)
	       g.drawImage(temp,10,10,temp.getWidth(),temp.getHeight(), this);  
	  }  
	}
////////////////////////////////////////////////////////////////////////

public class Main {
  private static final int THRESH_BINARY = 0;

public static void main(String[] args) {
    System.out.println("Hello, OpenCV");
    
    // Load the native library.
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

    // Load first image
    Mat image0 = Highgui.imread("D:/SOFT/eclipse_keplerClassic/workspace/java_cv_ImageDifference/imageDifference0.png");
    // Transform first image to gray colour image
    Mat image0Gray = new Mat(); // at first we create this gray img
    Imgproc.cvtColor(image0, image0Gray, Imgproc.COLOR_BGR2GRAY); //then transform

    // Load second image
    Mat image1 = Highgui.imread("D:/SOFT/eclipse_keplerClassic/workspace/java_cv_ImageDifference/imageDifference1.png");
    // Transform first image to gray colour image
    Mat image1Gray = new Mat(); // at first we create this gray img
    Imgproc.cvtColor(image1, image1Gray, Imgproc.COLOR_BGR2GRAY); //then transform
    
    // Find the difference from images
    Mat imageDifference = new Mat();
    Core.absdiff(image0Gray, image1Gray, imageDifference);
    
    // Get threshold img
    Mat imageThreshold = new Mat();
    Imgproc.threshold(imageDifference, imageThreshold, 20, 255, THRESH_BINARY);
    
    // Save the visualized detection.
    String filename = "imageThreshold.png";
    System.out.println(String.format("Writing %s", filename));
    Highgui.imwrite(filename, imageThreshold);
    
    
    // Consider the image for processing
    Mat image = Highgui.imread("D:/SOFT/eclipse_keplerClassic/workspace/java_cv_ImageDifference/imageDifference1.png", Imgproc.COLOR_BGR2GRAY);
    Mat imageHSV = new Mat(image.size(), Core.DEPTH_MASK_8U);
    Mat imageBlurr = new Mat(image.size(), Core.DEPTH_MASK_8U);
    Mat imageA = new Mat(image.size(), Core.DEPTH_MASK_ALL);
    Imgproc.cvtColor(image, imageHSV, Imgproc.COLOR_BGR2GRAY);
    Imgproc.GaussianBlur(imageHSV, imageBlurr, new Size(5,5), 0);
    Imgproc.adaptiveThreshold(imageBlurr, imageA, 255,Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY,7, 5);

    Highgui.imwrite("D:/SOFT/eclipse_keplerClassic/workspace/java_cv_ImageDifference/imageBlurr.png",imageBlurr);

    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();    
    Imgproc.findContours(imageA, contours, new Mat(), Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);
    //Imgproc.drawContours(imageBlurr, contours, 1, new Scalar(0,0,255));
    for(int i=0; i< contours.size();i++){
        System.out.println("Imgproc.contourArea: " + Imgproc.contourArea(contours.get(i)));
        if (Imgproc.contourArea(contours.get(i)) > 50 ){
            Rect rect = Imgproc.boundingRect(contours.get(i));
            //System.out.println(rect.height);
            if (rect.height > 28){
            System.out.println("(rect.XY, rect.HW: " + rect.x +","+rect.y+","+rect.height+","+rect.width);
            Core.rectangle(image, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height),new Scalar(0,0,255));
            }
        }
    }
    Highgui.imwrite("D:/SOFT/eclipse_keplerClassic/workspace/java_cv_ImageDifference/result.png",image);
    
    /*
    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    Mat hierarchy = new Mat();
    Mat contoursFrame = new Mat();
    contoursFrame = imageThreshold.clone();
    Imgproc.findContours(contoursFrame, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
    int contoursCounter = contours.size();
    Imgproc.cvtColor(contoursFrame, contoursFrame, Imgproc.COLOR_GRAY2BGR);
    
    Imgproc.drawContours(contoursFrame, contours, -1, new Scalar(255, 255, 255), 1); //Everything
    Imgproc.drawContours(contoursFrame, contours, 1, new Scalar(0, 255, 0), 2); //#1 square (green)
    Imgproc.drawContours(contoursFrame, contours, 6, new Scalar(0, 255, 255), 2); //#2 square (yellow)
    Imgproc.drawContours(contoursFrame, contours, 3, new Scalar(0, 0, 255), 2); //#3 square (red)
    Imgproc.drawContours(contoursFrame, contours, 4, new Scalar(255, 0, 0), 2); //#4 square (blue)
    
    MatOfPoint2f approxCurve = new MatOfPoint2f();

    //For each contour found
    for (int i=0; i<contours.size(); i++)
    {
        //Convert contours(i) from MatOfPoint to MatOfPoint2f
        MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(i).toArray() );
        //Processing on mMOP2f1 which is in type MatOfPoint2f
        double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;
        Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

        //Convert back to MatOfPoint
        MatOfPoint points = new MatOfPoint( approxCurve.toArray() );

        // Get bounding rect of contour
        Rect rect = Imgproc.boundingRect(points);

        // draw enclosing rectangle (all same color, but you could use variable i to make them unique)
        //Core.rectangle(contoursFrame, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), (255, 0, 0, 255), 3); 
       
        // draw enclosing rectangle (all same color, but you could use variable i to make them unique)
        //Point(rect.x+rect.width,rect.y+rect.height) , new Scalar(255, 0, 0, 255), 3); 
    }
    */


    
    trackGreen();
    
  }



////////////////////////////////////////////////////////////////////////////////////////////////////////////
private static void trackGreen(){
    JFrame frame1 = new JFrame("Camera");  
      frame1.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);  
      frame1.setSize(640,480);  
      frame1.setBounds(0, 0, frame1.getWidth(), frame1.getHeight());  
      Panel panel1 = new Panel();  
      frame1.setContentPane(panel1);  
      frame1.setVisible(true);  
      JFrame frame2 = new JFrame("HSV");  
      frame2.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);  
      frame2.setSize(640,480);  
      frame2.setBounds(300,100, frame2.getWidth()+300, 100+frame2.getHeight());  
      Panel panel2 = new Panel();  
      frame2.setContentPane(panel2);  
      frame2.setVisible(true);  
      JFrame frame4 = new JFrame("Threshold");  
      frame4.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);  
      frame4.setSize(640,480);  
      frame4.setBounds(900,300, frame2.getWidth()+900, 300+frame2.getHeight());  
      Panel panel4 = new Panel();  
      frame4.setContentPane(panel4);      
      frame4.setVisible(true);  
      //-- 2. Read the video stream  
      VideoCapture capture =new VideoCapture(0); 
      //capture.set(15, 30);
      Mat webcam_image=new Mat();  
      Mat hsv_image=new Mat();  
      Mat thresholded=new Mat();  
      Mat thresholded2=new Mat();  
       capture.read(webcam_image);  
       frame1.setSize(webcam_image.width()+40,webcam_image.height()+60);  
       frame2.setSize(webcam_image.width()+40,webcam_image.height()+60);  
       //frame3.setSize(webcam_image.width()+40,webcam_image.height()+60);  
       frame4.setSize(webcam_image.width()+40,webcam_image.height()+60);  
      Mat array255=new Mat(webcam_image.height(),webcam_image.width(),CvType.CV_8UC1);  
      array255.setTo(new Scalar(255));  
      /*Mat S=new Mat();  
      S.ones(new Size(hsv_image.width(),hsv_image.height()),CvType.CV_8UC1);  
      Mat V=new Mat();  
      V.ones(new Size(hsv_image.width(),hsv_image.height()),CvType.CV_8UC1);  
          Mat H=new Mat();  
      H.ones(new Size(hsv_image.width(),hsv_image.height()),CvType.CV_8UC1);*/  
      Mat distance=new Mat(webcam_image.height(),webcam_image.width(),CvType.CV_8UC1);  
      //new Mat();//new Size(webcam_image.width(),webcam_image.height()),CvType.CV_8UC1);  
      List<Mat> lhsv = new ArrayList<Mat>(3);      
      Mat circles = new Mat(); // No need (and don't know how) to initialize it.  
                   // The function later will do it... (to a 1*N*CV_32FC3)  
      Scalar hsv_min = new Scalar(30, 100, 100, 0);  
      Scalar hsv_max = new Scalar(55, 255, 255, 0);  
      Scalar hsv_min2 = new Scalar(60, 100, 100, 0);  
      Scalar hsv_max2 = new Scalar(90, 255, 255, 0);  
      double[] data=new double[3];  
      
      Mat image0 = new Mat();
      Mat image1 = new Mat();
      Mat image0Gray = new Mat();
      Mat image1Gray = new Mat();
      Mat imageDifference = new Mat();
      Mat imageThreshold = new Mat();
      Mat image0F = new Mat();
      
      Mat image = new Mat();
      Mat imageHSV = new Mat();
      Mat imageBlurr = new Mat();
      Mat imageA = new Mat();
 
      if( capture.isOpened())  
      {  
       while( true )  
       {  
         capture.read(webcam_image); 
         capture.read(image0);
         capture.read(image1);
         if( !webcam_image.empty() )  
         {  
          // One way to select a range of colors by Hue  
          Imgproc.cvtColor(webcam_image, hsv_image, Imgproc.COLOR_BGR2HSV);  
          Core.inRange(hsv_image, hsv_min, hsv_max, thresholded);           
          Core.inRange(hsv_image, hsv_min2, hsv_max2, thresholded2);
           Core.bitwise_or(thresholded, thresholded2, thresholded); 
           Imgproc.erode(thresholded, thresholded, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8,8)));
           Imgproc.dilate(thresholded, thresholded, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8, 8)));
           // Notice that the thresholds don't really work as a "distance"  
          // Ideally we would like to cut the image by hue and then pick just  
          // the area where S combined V are largest.  
          // Strictly speaking, this would be something like sqrt((255-S)^2+(255-V)^2)>Range  
          // But if we want to be "faster" we can do just (255-S)+(255-V)>Range  
          // Or otherwise 510-S-V>Range  
          // Anyhow, we do the following... Will see how fast it goes...  

           //Transform first and second frame into the grayimages
           Imgproc.cvtColor(image0, image0Gray, Imgproc.COLOR_BGR2GRAY); //then transform
           Imgproc.cvtColor(image1, image1Gray, Imgproc.COLOR_BGR2GRAY); //then transform
           
           //Find the difference between
           Core.absdiff(image0Gray, image1Gray, imageDifference);
           
           //Threshold
           Imgproc.threshold(imageDifference, imageThreshold, 20, 255, THRESH_BINARY);
           
           Imgproc.blur(imageThreshold, imageThreshold, new Size(10, 10));
           
           Imgproc.threshold(imageDifference, imageThreshold, 20, 255, THRESH_BINARY);
           
           //Draw rechtangle above the contors
           //http://answers.opencv.org/question/12056/findcontours-in-java-not-giving-desired-results/
           List<MatOfPoint> contours = new ArrayList<MatOfPoint>();    
           Imgproc.findContours(imageThreshold, contours, new Mat(), Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);
           //Imgproc.drawContours(imageBlurr, contours, 1, new Scalar(0,0,255));
           for(int i=0; i< contours.size();i++){
               System.out.println("Imgproc.contourArea: " + Imgproc.contourArea(contours.get(i)));
               if (Imgproc.contourArea(contours.get(i)) > 50 ){
                   Rect rect = Imgproc.boundingRect(contours.get(i));
                   //System.out.println(rect.height);
                   if (rect.height > 28){
                   System.out.println("(rect.XY, rect.HW: " + rect.x +","+rect.y+","+rect.height+","+rect.width);
                   Core.rectangle(image0, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height),new Scalar(0,0,255));
                   }
               }
           }
           Highgui.imwrite("D:/SOFT/eclipse_keplerClassic/workspace/java_cv_ImageDifference/imageBlurr.png",image0);
          ///////////////////////

          //-- 5. Display the image  
          Core.flip(image0, image0F, 0);
          Core.flip(image0F, image0F, 1);
          panel1.setimagewithMat(image0F);  
          panel2.setimagewithMat(imageDifference);  
          panel4.setimagewithMat(imageThreshold);  
          frame1.repaint();  
          frame2.repaint();   
          frame4.repaint();  
         }  
         else  
         {  
           System.out.println(" --(!) No captured frame -- Break!");  
           break;  
         } 
         }  
        }  
}
/*
////////////////////////////////////
private static void searchForMovement(Mat thresholdImage, Mat cameraFeed){
//notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
//to take the values passed into the function and manipulate them, rather than just working with a copy.
//eg. we draw to the cameraFeed to be displayed in the main() function.
boolean objectDetected = false;
Mat temp;
thresholdImage.copyTo(temp);
//these two vectors needed for output of findContours
vector< vector<Point> > contours;
Vector objects = new Vector();
vector<Vec4i> hierarchy;
//find contours of filtered image using openCV findContours function
//findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours

//if contours vector is not empty, we have found some objects
if(contours.size()>0)objectDetected=true;
else objectDetected = false;

if(objectDetected){
//the largest contour is found at the end of the contours vector
//we will simply assume that the biggest contour is the object we are looking for.
vector< vector<Point> > largestContourVec;
largestContourVec.push_back(contours.at(contours.size()-1));
//make a bounding rectangle around the largest contour then find its centroid
//this will be the object's final estimated position.
objectBoundingRectangle = boundingRect(largestContourVec.at(0));
int xpos = objectBoundingRectangle.x+objectBoundingRectangle.width/2;
int ypos = objectBoundingRectangle.y+objectBoundingRectangle.height/2;

//update the objects positions by changing the 'theObject' array values
theObject[0] = xpos , theObject[1] = ypos;
}
//make some temp x and y variables so we dont have to type out so much
int x = theObject[0];
int y = theObject[1];

//draw some crosshairs around the object
circle(cameraFeed,Point(x,y),20,Scalar(0,255,0),2);
line(cameraFeed,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
line(cameraFeed,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
line(cameraFeed,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
line(cameraFeed,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);

//write the position of the object to the screen
putText(cameraFeed,"Tracking object at (" + intToString(x)+","+intToString(y)+")",Point(x,y),1,1,Scalar(255,0,0),2);



}
*/
////////////////////////////////////


}




