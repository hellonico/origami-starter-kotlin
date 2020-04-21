package origami.kotlin

import org.opencv.core.*
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import origami.Camera
import origami.Dnn
import origami.Origami
import java.util.function.Function

object HelloCam {
    @JvmStatic
    fun main(args: Array<String>) {
        Origami.init()
        val list = Dnn.readNetFromSpec("networks.caffe:convnet-age:1.0.0")
        val _net: Net = list.get(0) as Net
        val labels = list.get(2) as List<*>

        val e = Function { f: Mat ->
            val g = Mat()
            Imgproc.resize(f, g, Size(((f.width() / 2).toDouble()), (f.height() / 2).toDouble()))
            g
        }
        Camera().device(0).filter(e).run()
    }
}