//
//  MeasurementView.swift
//  YoloV8
//
//  Created by csc-wt on 8/6/24.
//

import Foundation
import SwiftUI

struct PhotoPopupView: View {
    // The photo to display in the popup
    let photo: UIImage?

    var body: some View {
        VStack {
            if let photo = photo {
                Image(uiImage: photo)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 300, height: 300)
            } else {
                Text("No photo available")
            }
            
            Button(action: {
                // Dismiss the popup
                // In SwiftUI 3.0 and later, you can use the environment dismiss action
                UIApplication.shared.windows.first?.rootViewController?.dismiss(animated: true, completion: nil)
            }) {
                Text("Close")
                    .padding()
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
        }
        .padding()
    }
}
