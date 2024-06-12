---
layout: post
title: "Color Matching Via Histogram Backprojection"
mathjax: true
categories: computer_vision
tag: computer_vision
---

# 1. Motivation 

## 1.1 Modern methods

These days object detection is typically done by machinelearning-based methods.
The backbone of these methods are Convolutional Neural Networks (CNNs).
Such methods are typically divided in one-stage and multiple-stage ones and some popular
ones are YOLO, SSD and faster RCNN and its derivatives. Let's look at them briefly to
show how fundamentally differenly they are to the classical method that I will present.

<table style="width: 100%; border-collapse: collapse;">
    <tr>
        <th style="border: 1px solid black; padding: 8px; background-color: #f2f2f2;">Name</th>
        <th style="border: 1px solid black; padding: 8px; background-color: #f2f2f2;">Stages</th>
        <th style="border: 1px solid black; padding: 8px; background-color: #f2f2f2;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid black; padding: 8px; text-align: center;">YOLO</td>
        <td style="border: 1px solid black; padding: 8px; text-align: center;">One</td>
        <td style="border: 1px solid black; padding: 8px; text-align: center;">
        Divide an image in a grid. For each cell, describe it with (1) a class probability and
        (2) some bounding boxes with height, width, centre and confidence. Then, non-max suppression
        to compute the Intersection of Union (IoU) and reject overlapping boxes and keep only the
        one with highest confidence. </td>
    </tr>
    <tr>
        <td style="border: 1px solid black; padding: 8px; text-align: center;">SSD</td>
        <td style="border: 1px solid black; padding: 8px; text-align: center;">One</td>
        <td style="border: 1px solid black; padding: 8px; text-align: center;">Assigns class
        probabilities similarly to YOLO. The difference is that it performs detection at
        multiple resolutions and produces bounding boxes of fixed size (anchors) at each.
        Sums the multi-res maps and compares the IoU to a threshold to derive final
        bounding boxes.</td>
    </tr>
    <tr> 
        <td style="border: 1px solid black; padding: 8px; text-align: center;">faster RCNN</td>
        <td style="border: 1px solid black; padding: 8px; text-align: center;">Many</td>
        <td style="border: 1px solid black; padding: 8px; text-align: center;">Selective
        search to draw boxes around object candidates. Warp each region around a fixed size
        and pass it to a CNN which computes features. Feed features to a classifier. Non-
        max suppression to eliminate redundant boxes.</td>
    </tr>
</table>

These are pretty accurate and work in real time. However, they require high computational 
power. A lot of modern approaches struggle to run in real-time without CPUs specialised in AI.
Getting them to run can also be a pain because of all the libraries they require (and trust me, 
I've been through that pain a lot). This makes them unsuitable for low-power devices.

## 1.2 The idea of matching via histograms

This is where a very elegant and lightweight approach from the 90s by M. Swain 
and D. Ballard comes in handy, and that is histogram backprojection. As the 
name suggests, this relies on colour operations (histogram) as opposed to point 
operations (pixel locations). It cannot do full object detection on its own, but 
it's very good at matching two images.

# 2. Histogram backprojection

## 2.1 Input/Output

The input is an RGB image $I_{m \times n}$ and a model $M$.  The histogram
of the model describes what we want to match at every region of the
image $I$. The algorithm outputs an image $b_{m\times n} \in [0,1]$.

The closer each value of $b$ to 1, the better the match.

## 2.2 Histogram ratio

The backprojection method relies on the histogram ratio as a matching criterion between
two images to let's define it as $R$:

$R = \frac{H_M}{H_I}$

, where $H\_M, H\_I$ are the histograms of the model and image respectively.

To demonstrate what the ratio means for the two images, we will only consider the histogram
of the red channel. Therefore $H_M$, $H_I$ are 255-length vectors. We will also normalise
them.
In general $R$ is a matrix and $M, I$ are never greyscale but to make it intuitive assume
that they are, then $R$ is a 255-length vector. We will try to understand what the values
of $R$ mean.

<center>
<img src="https://raw.githubusercontent.com/leonmavr/leonmavr.github.io/master/_posts/2023-12-17-Observer-design-pattern/obs_one_to_many.png" alt="observer one-to-many" width="100%"/>
</center>
