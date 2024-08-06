//
//  YoloClassifier.swift
//  YoloV8
//
//

import Foundation
import CoreML
import Vision

// TODO: we only use UIkit for UIImage... is there a SwiftUI replacement?
import UIKit



struct ImageSegmentation {
    // TODO: change mask type to CVPixelBuffer?
    var mask: [[Float]]
    var boundingBox: CGRect
}

class ImageSegmenter {
    
    static func segmentImage(imgBuffer: CVPixelBuffer) -> ImageSegmentation {
        // Get width and height of imgBuffer
        let width = CVPixelBufferGetWidth(imgBuffer)
        let height = CVPixelBufferGetHeight(imgBuffer)
        
        // TODO: can return nil!
        let results = classifyImage(imgBuffer: imgBuffer)!
        
        let boundingBoxOut = getBoundingBox(feature:results.var_1550)
        let mask = getBestMask(boundingBoxOut, masks: results.p, ogWidth: width, ogHeight: height)
        
        return ImageSegmentation(mask: mask, boundingBox: boundingBoxOut.boundingBox)
    }
    
    
    static func segmentImageVS(image: UIImage, onSegmented: @escaping ([Any])->Void) {
        // TODO: implement
        classifyVisionModel(onClassified: onSegmented)
    }
    
    
    private struct BoundingBoxOutput {
        var boundingBox: CGRect
        var maxMaskIdx: Int
    }
    
    
    static private func getBoundingBox(feature: MLMultiArray) -> BoundingBoxOutput {
        var boundingBox = CGRect(x: 0, y: 0, width: 10, height: 10)
        
        var probMaxIdx = 0
        var maxProb : Float = 0
        var box_x : Float = 0
        var box_y : Float = 0
        var box_width : Float = 0
        var box_height : Float = 0
        
        for j in 0..<feature.shape[2].intValue-1
        {
            let key = [0,4,j] as [NSNumber]
            let nextKey = [0,4,j+1] as [NSNumber]
            if(feature[key].floatValue < feature[nextKey].floatValue){
                if(maxProb < feature[nextKey].floatValue){
                    probMaxIdx = j+1
                    let xKey = [0,0,probMaxIdx] as [NSNumber]
                    let yKey = [0,1,probMaxIdx] as [NSNumber]
                    let widthKey = [0,2,probMaxIdx] as [NSNumber]
                    let heightKey = [0,3,probMaxIdx] as [NSNumber]
                    maxProb = feature[nextKey].floatValue
                    box_width = feature[widthKey].floatValue
                    box_height = feature[heightKey].floatValue
                    
                    box_x = feature[xKey].floatValue - (box_width/2)
                    box_y = feature[yKey].floatValue - (box_height/2)
                }
            }
        }
        boundingBox = CGRect(x: CGFloat(box_x)
                             ,y: CGFloat(box_y)
                             ,width: CGFloat(box_width)
                             ,height: CGFloat(box_height))
        var maxMaskProb : Float = 0
        var maxMaskIdx = 0
        
        // CSC: TODO: why are we only looking at first 5 indeces?
        for maskPrbIdx in 5..<feature.shape[1].intValue-1{
            let key = [0, maskPrbIdx, probMaxIdx] as [NSNumber]
            let nextKey = [0, maskPrbIdx + 1, probMaxIdx] as [NSNumber]
            if(feature[key].floatValue < feature[nextKey].floatValue){
                if(maxMaskProb < feature[nextKey].floatValue){
                    maxMaskIdx = maskPrbIdx + 1
                    maxMaskProb = feature[nextKey].floatValue
                }
            }
            let bestMaskIdx = maxMaskIdx-5
            // Swift.print("\(maskPrbIdx-5) Best mask probablity is \(maxMaskIdx-5) with value \(maxMaskProb)")
        }
        
        // Swift.print("Bounding box from classifier \(boundingBox)")
        return BoundingBoxOutput(boundingBox: boundingBox, maxMaskIdx: maxMaskIdx)
    }
    
    
    static private func getBestMask(_ boundingBoxOut: BoundingBoxOutput, masks: MLMultiArray, ogWidth: Int, ogHeight: Int) -> [[Float]] {
        //let testImage = UIImage(contentsOfFile: Bundle.main.path(forResource: "tree-test", ofType: "png")!)!
        let boundingBox = boundingBoxOut.boundingBox
        let bestMaskIdx = boundingBoxOut.maxMaskIdx
        
        // TODO: do we need to divide by 2 like in the orignal code?
        let width = CGFloat(ogWidth) // /2
        let height = CGFloat(ogHeight) // /2
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
        
        let scaledX : CGFloat = (boundingBox.minX/1024)*width
        let scaledY : CGFloat = (boundingBox.minY/1024)*height
        let scaledWidth : CGFloat = (boundingBox.width/1024)*width
        let scaledHeight : CGFloat = (boundingBox.height/1024)*height
        let rectangle = CGRect(x: scaledX, y: scaledY, width: scaledWidth, height: scaledHeight)
    
        print("scaled rectangle \(rectangle)")
        
        
        let maskProbThreshold : Float = 0.5
        let maskFill : Float = 1.0
        
        //draw the mask
        var maskProbalities : [[Float]] = [] //this will contains 160x160 mask pixel probablities
        var maskProbYAxis : [Float] = []
        
        print("Actual Image bounds \(rectangle)")
        
        //get the bounds for mask to match the bounds
        let mask_x_min = (rectangle.minX/width)*256
        let mask_x_max = (rectangle.maxX/width)*256
        
        let mask_y_min = (rectangle.minY/height)*256
        let mask_y_max = (rectangle.maxY/height)*256
        
        for y in 0..<masks.shape[2].intValue {
            maskProbYAxis.removeAll()
            for x in 0..<masks.shape[3].intValue {
                let pointKey = [0,bestMaskIdx,y,x] as [NSNumber]
                if(sigmoid(z: masks[pointKey].floatValue) < maskProbThreshold
                   && x >=  Int(mask_x_min) && x <= Int(mask_x_max)
                   && y >= Int(mask_y_min) && y <= Int(mask_y_max))
                {
                    maskProbYAxis.append(1.0)
                }
                else {
                    maskProbYAxis.append(0.0)
                }
            }
            maskProbalities.append(maskProbYAxis)
        }
        
       return maskProbalities
    }
    
    
    static private func sigmoid(z:Float) -> Float{
        return 1.0/(1.0+exp(z))
    }
    
    
    static private func classifyImage(imgBuffer: CVPixelBuffer) -> FastSAM_xOutput?{
        let fsModelWrapper = try? FastSAM_x()
        
        guard let fsModel = fsModelWrapper else{
            return nil
        }
        
        // let imageUrl = Bundle.main.url(forResource: "tree-test", withExtension: "png")!
        
        do{
            let output = try fsModel.prediction(input: FastSAM_xInput(image: imgBuffer))
            print(output)
            return output
        }
        catch{
            print("\(error)")
        }
       return nil
    }
    
    
    static private func classifyVisionModel(onClassified: @escaping ([Any])->Void) {
        let fsModelWrapper = try? FastSAM_x()
        
        guard let fsModel = fsModelWrapper else{
            return
        }
        do
        {
            let visionModel = try VNCoreMLModel(for: fsModel.model)
            let segmentationRequest = VNCoreMLRequest(model: visionModel, completionHandler: {(req,err) in
                if let results = req.results{
                    onClassified(results)
                }
            })
            let processingRequests = [segmentationRequest]
            let segmentationRequestHandler = VNImageRequestHandler(url: Bundle.main.url(forResource: "tree-test", withExtension: "png")!
                                                                   , orientation: .up)
            try segmentationRequestHandler.perform(processingRequests)
        }
        catch{
            print("\(error)")
        }
    }
}
