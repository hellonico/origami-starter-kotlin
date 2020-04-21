package origami.kotlin

import org.opencv.core.*
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import origami.Camera
import origami.Dnn
import origami.Origami
import java.util.function.Function

object AgeCam {
    @JvmStatic
    fun main(args: Array<String>) {
        Origami.init()
        val list = Dnn.readNetFromSpec("networks.caffe:convnet-age:1.0.0")
        val _net: Net = list.get(0) as Net
        val labels = list.get(2) as List<*>

        val age = Function { f: Mat ->
            val inputBlob = org.opencv.dnn.Dnn.blobFromImage(f, 1.0, Size(256.0, 256.0), Scalar(0.0), true, true)
            _net.setInput(inputBlob)
            _net.setPreferableBackend(org.opencv.dnn.Dnn.DNN_BACKEND_OPENCV)
            val result = _net.forward()

            val minmax = Core.minMaxLoc(result)
            Imgproc.putText(f, labels.get(minmax.maxLoc.x.toInt()) as String?, Point(100.0,100.0),1,2.0,Scalar(0.0,0.0,0.0));
            f
        }

        Camera().device(0).filter(age).run()
    }
}