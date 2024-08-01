//
//  YoloClassifier.swift
//  YoloV8
//
//

import Foundation
import CoreML
import Vision
class YoloClassifier{
    
    
    
    func classifyImage() -> FastSAM_xOutput?{
        let fsModelWrapper = try? FastSAM_x()
        
        guard let fsModel = fsModelWrapper else{
            return nil
        }
        
        let imageUrl = Bundle.main.url(forResource: "tree-test", withExtension: "png")!
        
        do{
            let output = try fsModel.prediction(input: FastSAM_xInput(imageAt: imageUrl))
            print(output)
            return output
        }
        catch{
            print("\(error)")
        }
       return nil
    }
    
    func classifyVisionModel(onClassified: @escaping ([Any])->Void){
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
