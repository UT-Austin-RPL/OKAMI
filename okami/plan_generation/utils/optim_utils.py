import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from lietorch import SE3, SO3, LieGroupParameter

class LWRRegressor:
    """This is an implementation of locally-weighted regression. This is used for estimating a trajectory. This is a very simplified version of DMP. 
    """
    def __init__(self):
        pass

    def weights_calculate(self, x0, X, tau):
        return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau **2) ))

    def regression(self, x0, X, Y, tau):
        x0 = np.c_[np.ones(len(x0)), x0]
        X = np.c_[np.ones(len(X)), X]
        
        # fit model: normal equations with kernel
        xw = X.T * self.weights_calculate(x0, X, tau)
        theta = np.linalg.pinv(xw @ X) @ xw @ Y
        # "@" is used to
        # predict value
        return x0 @ theta
    
    def predict_seq(self, x_seq, X, Y, tau):
        y_seq = []
        for x in x_seq:
            y_seq.append(self.regression([x], X, Y, tau)[0])
        return np.array(y_seq)

class SimpleBundleOptimization:
    """This is a very simple optimization that does the same thing as for simple bundle adjustment.

    This optimization takes in a temporal sequence of multiple keypoints, and estimate a sequence of transformation that can best describe the changes of keypoints.
    """
    def __init__(self):
        pass

    def optimize(self, 
                 all_keypoints, 
                 all_visibilities=None,
                 optim_kwargs={"lr": 0.001, 
                               "num_epochs": 2001,
                               "verbose": True}):
        transforms = [torch.eye(4, device="cuda", requires_grad=True) for _ in range(len(all_keypoints) - 1)]

        transform_centroid = torch.zeros(3, device="cuda", requires_grad=False).repeat(len(all_keypoints) - 1, 1)
        
        optimizer = optim.Adam(transforms, lr=optim_kwargs["lr"])
        num_epochs = optim_kwargs["num_epochs"]

        confidence_weights = torch.ones(all_keypoints.shape[0], all_keypoints.shape[1]).cuda()

        all_keypoints = all_keypoints.cuda()
        if all_visibilities is not None:
           # set confidence weights to 0.1 where all_visibilities are False
           confidence_weights = confidence_weights - (1 - all_visibilities.cuda().float()) * 0.9

        transforms = torch.stack(transforms, dim=0)
        # Optimization loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(transforms, transform_centroid, all_keypoints, confidence_weights)
            loss.backward()
            optimizer.step()

            if optim_kwargs["verbose"] and epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        print(f"Final loss: {loss.item()}")
        optimized_transforms = [t.detach().cpu().numpy() for t in transforms]
        optimized_centroids = [t.detach().cpu().numpy() for t in transform_centroid]
        return optimized_transforms# , optimized_centroids

    def apply_homogeneous_transform(self, points, transform):
        # Apply SE(3) transformation in PyTorch
        R, t = transform[:3, :3], transform[:3, 3]
        return torch.matmul(R, points.T) + t[:, None]
    
    def compute_loss(self, transforms, transform_centroid, all_keypoints, confidence_weights):
        M = len(all_keypoints) - 1
        loss = 0.0

        # Reprojection error
        transformed_points = torch.vmap(self.apply_homogeneous_transform)(all_keypoints[:-1], transforms)
        transformed_points = transformed_points.permute(0, 2, 1)
        loss += torch.norm((all_keypoints[1:] - transformed_points) * confidence_weights[1:][..., None], p=2)
        # Smoothness constraint
        loss += 0.1 * torch.norm(transforms[:-1, :3, :3] - transforms[1:, :3, :3]) # Rotation difference
        loss += 0.1 * torch.norm(transforms[:-1, :3, 3] - transforms[1:, :3, 3]) # Translation difference
        return loss    


class QuatBundleOptimization:
    def __init__(self):
        pass

    def optimize(self, 
                 all_keypoints, 
                 all_visibilities=None,
                 optim_kwargs={"lr": 0.001, "num_epochs": 2001, "verbose": True}):
        # Initialize quaternions and translations
        # quaternions = [torch.tensor([1., 0., 0., 0.], requires_grad=True) for _ in range(len(all_keypoints) - 1)]
        quaternions = [torch.rand(4, requires_grad=True) for _ in range(len(all_keypoints) - 1)]

        translations = [torch.rand(3, requires_grad=True) for _ in range(len(all_keypoints) - 1)]

        optimizer = optim.Adam(quaternions + translations, lr=optim_kwargs["lr"])
        num_epochs = optim_kwargs["num_epochs"]
        # transform_centroid = [torch.randn(3, requires_grad=True) for _ in range(len(all_keypoints) - 1)]
        transform_centroid = [torch.zeros(3, requires_grad=False) for _ in range(len(all_keypoints) - 1)]
        confidence_weights = torch.ones(all_keypoints.shape[0], all_keypoints.shape[1])
        if all_visibilities is not None:
            confidence_weights = confidence_weights - (1 - all_visibilities.float()) * 0.9

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(quaternions, translations, transform_centroid, all_keypoints, confidence_weights)
            loss.backward()
            optimizer.step()

            if optim_kwargs["verbose"] and epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        print(f"Final loss: {loss.item()}")
        optimized_rotation = [self.quaternion_to_matrix(q).detach().numpy() for q in quaternions]
        optimized_quaternions = [q.detach().numpy() for q in quaternions]
        print(optimized_quaternions)
        optimized_translations = [t.detach().numpy() for t in translations]
        optimized_transformation = []
        for rotation, translation in zip(optimized_rotation, optimized_translations):
            transformation = np.eye(4)
            transformation[:3, :3] = rotation
            transformation[:3, 3] = translation
            optimized_transformation.append(transformation)
        return optimized_transformation# , transform_centroid

    def quaternion_to_matrix(self, quaternion):
        """ Convert a quaternion into a rotation matrix. """
        q = F.normalize(quaternion.unsqueeze(-1), dim=0)  # Normalize the quaternion
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        # Compute rotation matrix
        return torch.tensor([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])

    def apply_homogeneous_transform(self, points, quaternion, translation):
        """ Apply SE(3) transformation using quaternion and translation. """
        R = self.quaternion_to_matrix(quaternion)
        return torch.matmul(R, points.T) + translation[:, None]

    def compute_loss(self, quaternions, translations, transform_centroid, all_keypoints, confidence_weights):
        M = len(all_keypoints) - 1
        loss = 0.0

        # Reprojection error
        for i in range(M):
            transformed_points = self.apply_homogeneous_transform(all_keypoints[i], quaternions[i], translations[i])
            loss += torch.norm(((all_keypoints[i+1]).T - transformed_points) * confidence_weights[i], p=2)

        # Smoothness constraint
        for i in range(1, M):
            loss += torch.norm(translations[i] - translations[i-1])
            loss += torch.norm(quaternions[i] - quaternions[i-1])  # This is a simple approximation

        return loss


class LieBundleOptimization:
    """This is a very simple optimization that does the same thing as for simple bundle adjustment.

    This optimization takes in a temporal sequence of multiple keypoints, and estimate a sequence of transformation that can best describe the changes of keypoints.
    """
    def __init__(self):
        pass

    def optimize(self, 
                 all_keypoints, 
                 all_visibilities=None,
                 regularization_weight_pos=1.0,
                 regularization_weight_rot=1.0,
                 optim_kwargs={"lr": 0.01, 
                               "num_epochs": 2001,
                               "verbose": True,
                               "momentum": 0.9}):


        # qxyz = torch.zeros(len(all_keypoints) - 1, 3, device="cuda", requires_grad=True)
        # qw = torch.ones(len(all_keypoints) - 1, 1, device="cuda", requires_grad=True)

        # q = torch.cat([qxyz, qw], dim=-1)
        q = torch.randn(len(all_keypoints) - 1, 4, device="cuda", requires_grad=True) * 0.1
        q = q / q.norm(dim=-1, keepdim=True)
        t = torch.zeros(len(all_keypoints) - 1, 3, device="cuda", requires_grad=True)
        
        R = SO3.InitFromVec(q)
        R = LieGroupParameter(R)

        if "momentum" not in optim_kwargs:
            optim_kwargs["momentum"] = 0.9

        num_epochs = optim_kwargs["num_epochs"]
        all_keypoints = torch.tensor(all_keypoints).float().cuda()
        if all_visibilities is not None:
            all_visibilities = torch.tensor(all_visibilities).float().cuda()

        confidence_weights = torch.ones(all_keypoints.shape[0], all_keypoints.shape[1]).cuda()

        if all_visibilities is not None:
           # set confidence weights to 0.1 where all_visibilities are False
           confidence_weights = confidence_weights - (1 - all_visibilities.float()) * 0.9
        optimizer = torch.optim.SGD([R],
                                    lr=optim_kwargs["lr"], 
                                    momentum=optim_kwargs["momentum"])
        # Optimization loop
        for epoch in range(num_epochs):

            optimizer.zero_grad()

            for param_group in optimizer.param_groups:
                param_group['lr'] = optim_kwargs["lr"] * .995**epoch

            loss = self.compute_rotation_loss(R, t, 
                                     all_keypoints, 
                                     confidence_weights, 
                                     regularization_weight_pos,
                                     regularization_weight_rot)
            loss.backward()
            optimizer.step()

            if optim_kwargs["verbose"] and epoch % 200 == 0:
                print(f"Rotation Epoch {epoch}, Loss: {loss.item()}")
        optimizer = torch.optim.SGD([R, t],
                                    lr=optim_kwargs["lr"], 
                                    momentum=optim_kwargs["momentum"])
        # Optimization loop

        minimum_loss = 1e10
        for epoch in range(num_epochs):

            optimizer.zero_grad()

            for param_group in optimizer.param_groups:
                param_group['lr'] = optim_kwargs["lr"] * .995**epoch

            loss = self.compute_refinement_loss(R, t, 
                                     all_keypoints, 
                                     confidence_weights, 
                                     regularization_weight_pos,
                                     regularization_weight_rot)
            loss.backward()
            optimizer.step()

            if loss.item() < minimum_loss:
                minimum_loss = loss.item()
                best_R = SO3.InitFromVec(R.group.data.detach().cpu()).matrix().detach().cpu().numpy()[..., :3, :3]
                best_t = t.detach().cpu().numpy()

            if optim_kwargs["verbose"] and epoch % 200 == 0:
                print(f"Refinement Epoch {epoch}, Loss: {loss.item()}")

        print(f"Final loss: {loss.item()}")

        result_R = SO3.InitFromVec(R.group.data.detach().cpu()).matrix().detach().cpu().numpy()[..., :3, :3]
        # return result_R, t.detach().cpu().numpy(), loss
        return best_R, best_t, loss
        # return result_T.detach().cpu().numpy() 

    def compute_rotation_loss(self, R, t, all_keypoints, confidence_weights, regularization_weight_pos=0.1, regularization_weight_ori=0.1, scaling=10.):
        M = len(all_keypoints) - 1
        loss = 0.0

        # print(R.shape, all_keypoints[:-1].shape, all_keypoints[:-1].mean(dim=1, keepdim=True).shape)
        estimated_points = R[:, None] * (all_keypoints[:-1] - all_keypoints[:-1].mean(dim=1, keepdim=True)) * scaling
        loss += (((estimated_points - (all_keypoints[1:] - all_keypoints[:-1].mean(dim=1, keepdim=True)) * scaling) * confidence_weights[1:].unsqueeze(-1)) ** 2).sum()

        R_diff = R[1:] * R[:-1].inv()
        regularization_loss = 0
        regularization_loss += regularization_weight_pos * (R_diff.log() ** 2).sum()
        loss += regularization_loss
        return loss

    def compute_refinement_loss(self, R, t, all_keypoints, confidence_weights, regularization_weight_pos=0.1, regularization_weight_ori=0.1):
        M = len(all_keypoints) - 1
        loss = 0.0

        new_R = R[:, None]
        estimated_points = new_R * (all_keypoints[:-1] - all_keypoints[:-1].mean(dim=1, keepdim=True)) + all_keypoints[:-1].mean(dim=1, keepdim=True) + t[:, None]
        loss += (((estimated_points - all_keypoints[1:]) * confidence_weights[1:].unsqueeze(-1)) ** 2).sum()

        R_diff = R[1:] * R[:-1].inv()
        regularization_loss = 0
        regularization_loss += regularization_weight_pos * (R_diff.log() ** 2).sum()
        loss += regularization_loss
        return loss

class NaiveSubgoalOptimization:
    """This is a very simple optimization that does the same thing as for simple bundle adjustment.

    This optimization takes in a temporal sequence of multiple keypoints, and estimate a sequence of transformation that can best describe the changes of keypoints.
    """
    def __init__(self):
        pass

    def optimize(self, 
                 all_keypoints, 
                 all_visibilities=None,
                 optim_kwargs={"lr": 0.1, 
                               "num_epochs": 201,
                               "verbose": True}):
        transforms = [torch.eye(4, requires_grad=True) for _ in range(len(all_keypoints) - 1)]

        transform_centroid = [torch.randn(3, requires_grad=True) for _ in range(len(all_keypoints) - 1)]
        optimizer = optim.Adam(transforms, lr=optim_kwargs["lr"])
        num_epochs = optim_kwargs["num_epochs"]

        all_keypoints = torch.tensor(all_keypoints).float()
        all_visibilities = torch.tensor(all_visibilities).float()

        confidence_weights = torch.ones(all_keypoints.shape[0], all_keypoints.shape[1])

        if all_visibilities is not None:
           # set confidence weights to 0.1 where all_visibilities are False
           confidence_weights = confidence_weights - (1 - all_visibilities.float()) * 0.9

        # Optimization loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(transforms, transform_centroid, all_keypoints, confidence_weights)
            loss.backward()
            optimizer.step()

            if optim_kwargs["verbose"] and epoch % 25 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        print(f"Final loss: {loss.item()}")
        optimized_transforms = [t.detach().numpy() for t in transforms]
        optimized_centroids = [t.detach().numpy() for t in transform_centroid]
        return optimized_transforms, optimized_centroids

    def apply_homogeneous_transform(self, points, transform):
        # Apply SE(3) transformation in PyTorch
        R, t = transform[:3, :3], transform[:3, 3]
        return torch.matmul(R, points.T) + t[:, None]
    
    def compute_loss(self, transforms, transform_centroid, all_keypoints, confidence_weights):
        M = len(all_keypoints) - 1
        loss = 0.0

        # Reprojection error
        for i in range(M):
            transformed_points = self.apply_homogeneous_transform(all_keypoints[i] - transform_centroid[i], transforms[i])
            loss += torch.norm(((all_keypoints[i+1] - transform_centroid[i]).T - transformed_points) * confidence_weights[i], p=2)

        # Smoothness constraint
        for i in range(1, M):
            loss += torch.norm(transforms[i][:3] - transforms[i-1][:3]) # Rotation difference
            loss += torch.norm(transforms[i][3:] - transforms[i-1][3:]) # Translation difference
        return loss 
    

class LieSubgoalOptimization:
    """This is a very simple optimization that does the same thing as for simple bundle adjustment.

    This optimization takes in a temporal sequence of multiple keypoints, and estimate a sequence of transformation that can best describe the changes of keypoints.
    """
    def __init__(self):
        pass

    def optimize(self, 
                 all_keypoints, 
                 all_visibilities=None,
                 optim_kwargs={"lr": 0.1, 
                               "num_epochs": 201,
                               "verbose": True,
                               "momentum": 0.9}):
        # transforms = [torch.eye(4, requires_grad=True) for _ in range(len(all_keypoints) - 1)]

        transform_centroid = [torch.randn(3, device="cuda", requires_grad=True) for _ in range(len(all_keypoints) - 1)]
        # optimizer = optim.Adam(transforms, lr=optim_kwargs["lr"])
        
        q = torch.randn(len(all_keypoints) - 1, 4, device="cuda", requires_grad=True)
        q = q / q.norm(dim=-1, keepdim=True)
        t = torch.randn(len(all_keypoints) - 1, 3, device="cuda", requires_grad=True)
        # transform_centroids = torch.randn(len(all_keypoints) - 1, 3, device="cuda", requires_grad=True)
        # centroid_q = torch.tensor([0., 0., 0., 1.], device="cuda", requires_grad=False).repeat(len(all_keypoints) - 1, 1)
        # centroid_q = torch.randn(len(all_keypoints) - 1, 4, device="cuda", requires_grad=True)
        # centroid_t = torch.randn(len(all_keypoints) - 1, 3, device="cuda", requires_grad=True)
        vec = torch.cat([t, q], dim=-1)
        T = SE3.InitFromVec(vec)
        T = LieGroupParameter(T)
        # centroid_T = SE3.InitFromVec(torch.cat([centroid_t, centroid_q], dim=-1))
        # centroid_T = LieGroupParameter(centroid_T)
        optimizer = torch.optim.SGD([T] + transform_centroid,
                                    lr=optim_kwargs["lr"], 
                                    momentum=optim_kwargs["momentum"])

        num_epochs = optim_kwargs["num_epochs"]

        all_keypoints = torch.tensor(all_keypoints).float().cuda()
        all_visibilities = torch.tensor(all_visibilities).float().cuda()

        confidence_weights = torch.ones(all_keypoints.shape[0], all_keypoints.shape[1]).cuda()

        if all_visibilities is not None:
           # set confidence weights to 0.1 where all_visibilities are False
           confidence_weights = confidence_weights - (1 - all_visibilities.float()) * 0.9

        # Optimization loop
        for epoch in range(num_epochs):

            optimizer.zero_grad()

            for param_group in optimizer.param_groups:
                param_group['lr'] = optim_kwargs["lr"] * .995**epoch

            loss = self.compute_loss(T, transform_centroid, all_keypoints, confidence_weights)
            loss.backward()
            optimizer.step()

            if optim_kwargs["verbose"] and epoch % 25 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        print(f"Final loss: {loss.item()}")
        # optimized_transforms = [t.detach().numpy() for t in transforms]
        # optimized_centroids = [t.detach().numpy() for t in transform_centroid]
        result_T = SE3.InitFromVec(T.group.data.detach().cpu()).matrix()
        
        return result_T.detach().cpu().numpy(), transform_centroid # optimized_transforms, optimized_centroids

    
    def compute_loss(self, T, transform_centroid, all_keypoints, confidence_weights):
        M = len(all_keypoints) - 1
        loss = 0.0
        for i in range(M):
            # transformed_points = T * (all_keypoints[i] - transform_centroid[i])
            # loss += torch.norm(((all_keypoints[i+1] - transform_centroid[i]) - transformed_points) * confidence_weights[i].reshape(-1, 1), p=2)
            transformed_points = T * (all_keypoints[i])
            loss += torch.norm(((all_keypoints[i+1]) - transformed_points) * confidence_weights[i].reshape(-1, 1), p=2)
        return loss