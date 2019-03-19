# Chapter 8 - Deploying Deep Learning Models to the Browser

In this chapter we will take the housing price prediction model that we imported at the end of the last chapter and will refactor it and embed it into an application to to make it ready for deployment to the web. We will make use of the TensorFlow.js sessions and will be using the `tidy` method to clean up our environment after our models have been run. This will help prevent memory leaks and ensure that variables created by our model runs are cleaned up appropriately. We will build a web server that will serve our application and models and will build a simple user interface that will allow users to interact with our model. So without further ado let's begin building our simple capstone project for this quick start guide.

## Building our Webserver

In order to perform deep learning in the browser we will need a way to host both our applications and models. To achieve this we will create a webserver to serve this purpose. Our webserver platform of choice will be Node.js. In the previous chapters we have used Node to train simple machine learning models; however, we will use it to build a server that will host the client-side code we will be writing in React.js. Our server will be written using a minimalist web development framework that is popular in the Node.js web developer community. This framework is called Express.js.

Before we beginning working on our webserver we should start by setting up our project.

TODO: COMPLETE THIS SECTION!

## Building the User Interface

TODO: REWRITE THIS ENTIRE SECTION!

## Importing the House Prediction Model into the Browser

TODO: REWRITE THIS ENTIRE SECTION!

## Sharding the Prediction Model for Delivery to the Browser

TODO: REWRITE THIS ENTIRE SECTION!

## Installing the Heroku CLI

Now that we have our server and client working locally on our machine we have to deploy it the web for consumption by users on the Internet. To this will have to deploy it to a server on the cloud. To make our lives easier we will use a tool called Heroku, that makes it very simple for us to achieve this task with very little setup. Let's start by signing up for the Heroku service by visiting http://heroku.com and clicking the "sign up" button in the top-right of the webpage and completing the sign up steps for a free account. You are free to sign up for a paid account if you like but this will not be required for the purposes of this tutorial.

Figure 8.x

![](./images/figure_8_x)

Next we will install the Heroku CLI, which is a command-line tool that will allow us to deploy our application to Heroku and interact it with it from a local terminal session. Heroku allows us to use simple commands that will allow us to deploy our application, restart it and even view the logs without having to physically remote into the machine. Since the installation instruction are different depending on the operating system you are using we will point you to https://devcenter.heroku.com/articles/heroku-cli to follow the most recent instructions required for installing the CLI on your computer.

After completing the installation steps, you may confirm that you are ready to move on to the next section by running the following command in a new terminal session.

```bash
heroku
```

The result should be the rendering of the man page for this command in your terminal, which should look something like the following output.

```bash
Usage: heroku COMMAND

Help topics, type heroku help TOPIC for more details:

 2fa
 access          manage user access to apps
 addons          tools and services for developing, extending, and operating your app
 apps            manage apps
 auth            heroku authentication
 authorizations  OAuth authorizations
 buildpacks      manage the buildpacks for an app
 certs           a topic for the ssl plugin
 ci              run an application test suite on Heroku
 clients         OAuth clients on the platform
 config          manage app config vars
 container       Use containers to build and deploy Heroku apps
 domains         manage the domains for an app
 drains          list all log drains
 features        manage optional features
 git             manage local git repository for app
 keys            manage ssh keys
 labs            experimental features
 local           run heroku app locally
 logs            display recent log output
 maintenance     manage maintenance mode for an app
 members         manage organization members
 notifications   display notifications
 orgs            manage organizations
 outbound-rules
 pg              manage postgresql databases
 pipelines
 plugins         add/remove CLI plugins
 ps              Client tools for Heroku Exec
 redis           manage heroku redis instances
 releases        manage app releases
 reviewapps
 run             run a one-off process inside a Heroku dyno
 sessions        OAuth sessions
 spaces          manage heroku private spaces
 status          status of the Heroku platform
 teams           manage teams
 twofactor
 webhooks        setup HTTP notifications of app activity
```

If you see this result, then congrats! You have successfully installed the Heroku CLI and are ready to move on to the next steps. The remaining sections of this chapter will assume that you have successfully completed the installation steps.

TODO: COMPLETE THIS SECTION!

### Deployment

TODO: REWRITE THIS ENTIRE SECTION!

## Basic Security Protection

Though this is not a production application, whenever we deploy a web application to the internet we want to add at least some level of security. One way to do this is to set various HTTP headers that will help protect our application from several attacks that can be made against our application while it's live on the web. Luckily for use members of the Node community have put together an Express middleware that will help protect use from several popular vulnerabilities. This middleware is called helmet and we can add add it our app as follows.

Let's add helmet as a dependency to our application by running the following command.

```bash
npx yarn add helmet
```

We can then add it to our app.js like so.

```javascript
const helmet = require('helemet');

...

app.use(helmet());
```

Now we can we will have the addition of several protections added to our app:

- Application of frameguard to prevent clickjacking
- Disabling of the browser's DNS prefetching by setting the X-DNS-Prefetch-Control header
- Hiding the X-Powered-By header that broadcasts to the world that you are using Express as your server-side framework
- Prevent the browser from serving your site over HTTP and to prefer HTTPS by setting the Strict-Transport-Security header
- Preventing cross-site scripting (XSS) attacks by setting the X-XSS-Protection header
- Disabling the sniffing of MIME types by setting the Content-Type-Options to nosniff
- Preventing malicious downloads in the context of your site in older versions of Internet Explorer by setting the X-Download-Options header to noopen

These settings won't guarantee full security of your application but they are a step in the right direction. If you would like to add more or learn more about these security threats, which is highly recommended, feel free to read the helmet documentation over at https://helmetjs.github.io/docs/.

## Interacting with Our App on the Internet

After deploying application we can easily view it on the internet by navigating to the Heroku URL we created in the browser. However, the Heroku CLI provides a convenience script that will allow us to easily do this from the command line. This command is as follows.

```bash
heroku open
```

By running the command above we should see our application loaded in the browser for further inspection. Everything should work the same way it did when we ran it locally. The only difference is that now we can view it on our desktop or mobile devices via the Internet. Congrats on a job well done! You have developed and deployed your first simple deep-learning powered web application. Now it's time to share the link with your family, friends and co-workers about how you earned your stripes as an aspiring machine learning engineer/web developer.

![]()

## Summary

In this chapter we deployed the house price prediction model that we created in the last chapter and made some improvements to ensure that our models run efficiently in a production environment. We used the `tidy` and `dispose` methods to clean up variables that were creating during the training process of our models.

TODO: COMPLETE THIS SECTION!

## Where to Go From Here

This chapter concludes our book. The aim of this book was to get you up to speed with TensorFlow.js and introduce you to enough machine learning and deep learning to get you productive as quickly as possible. This book was also designed to be used a quick reference that can be picked up to familiarize yourself on the topics covered. If you need to review the core API, then you may refer to that chapter, to review the Layers API refer to that chapter, to get an overview of deep learning in general you may reread that chapter and so on.

The next steps would be to continue building upon this example and adding more features to to it to make it more feature-rich. If you would like to dive deeper into the topics of machine learning and deep learning it is recommended to take a course or read a book that focuses more on the theory behind these subjects. This book purposes kept it light on theory so that we can focus on the TensforFlow.js API and provided enough theory to make it clear what the use case for TFJS is and what is actually happening behind the scenes. There are several university level courses and MOOCs that provide a very rigorous introduction to machine learning and deep learning theory and are available to audit for free online.

As web developers we more than likely will not be building these models, but will be embedding them into the applications that we build, in a way that is very similar to what we did in this chapter. However, the more we understand the mechanics of machine learning and deep learning the more value we can add in roles that require us to work with these models on a daily basis. So that models that we are using are not just a "black box" it's recommended to learn more about the different types of artificial neural networks and how they can be built using TensorFlow.

Another good way to learn about ANNs are to read academic papers and then try to build the models yourself. By reading papers you will learn which problems these ANNs are well-suited to solve. You will quickly learn that there are many artificial neural networks and that the problems they solve very much tailored to each ANN's use case but from here you may become inspired to dive deeper into the ones that solve the problems you're interested in or the ones that solve problems you see at work. One problem with implementing models that we read about in papers is the lack of data. For this I would recommend checking out Kaggle (https://www.kaggle.com/) for an extensive list of free data sets you can use for practice as well as data science competitions that you can compete in. Kaggle is also a good place to explore different techniques and algorithms used by members for inspiration.

Another way to understand what TensorFlow.js is doing underneath the hood is to get familiar with the TensorFlow Python library. Before diving into TFJS I worked with TensorFlow in Python and used it to embed TF models into webserver that I was building. When TensorFlow.js was announced I used it to import these same models into the browser to perform inference but it was my experience with the Python version of the library that made this transition seamless. Since we know that true power of TFJS lies in the ability to easily import existing models and them for inference and transfer learning being able to develop models in Python will widen the surface level of the types of models you can create since you will be able to use the extra computing power provided by the Python version of the library as well as leverage the power of several very powerful numerical and scientific computing libraries that have been built for Python.

With this being said, I hope that this book was able to serve its purpose as being a brief introduction to TensorFlow.js as well a quick reference for using its key functionality and core parts of its Core and Layers API. I would like to thank you again for taking the time to purchase and read this book. This has been my first ever book and this has been a very interesting journey to say the least. Good luck and God speed as you continue with your TensorFlow and deep learning journey!
