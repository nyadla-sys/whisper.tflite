import Foundation

class ModelDownloader: ObservableObject {
    @Published var progress: Double = 0
    @Published var isDownloading = false

    func download(url: String, fileName: String, completion: @escaping (URL?) -> Void) {
        let destDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let destURL = destDir.appendingPathComponent(fileName)

        if FileManager.default.fileExists(atPath: destURL.path) {
            completion(destURL)
            return
        }

        guard let remoteURL = URL(string: url) else { completion(nil); return }

        isDownloading = true
        progress = 0

        let task = URLSession.shared.downloadTask(with: remoteURL) { [weak self] tmpURL, _, error in
            DispatchQueue.main.async { self?.isDownloading = false }
            guard let tmpURL = tmpURL, error == nil else {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            try? FileManager.default.moveItem(at: tmpURL, to: destURL)
            DispatchQueue.main.async { completion(destURL) }
        }

        let observation = task.progress.observe(\.fractionCompleted) { [weak self] p, _ in
            DispatchQueue.main.async { self?.progress = p.fractionCompleted }
        }

        // Keep observation alive
        objc_setAssociatedObject(task, "obs", observation, .OBJC_ASSOCIATION_RETAIN)
        task.resume()
    }
}
