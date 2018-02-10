# ccminer_macos

This repository is forked from Tanguy Pruvot (tpruvot) to create a macOS version of ccminer to run easily on mac computers with NVidia eGPU's or Hackintosh.
Originally based on Christian Buchner's &amp; Christian H.'s CUDA project, no more active on github since 2014.

Check the [README.txt](README.txt) for the additions

Most work done on ccminer is done by others such as tpruvot so I'll just leave his BTC donation address here: 1AJdfCpLWPNoAMDfHF1wD5y8VgKSSTHxPo (tpruvot)

If you appreciate the macos specifics, you can use the following donation address for
ETH: 0x8FE5b2E81C265814e653574a4136480Fa294370d

A part of the recent algos were originally written by [djm34](https://github.com/djm34) and [alexis78](https://github.com/alexis78)

This variant was tested and built on MacOS (High Sierra) and runs using a NVidia GTX 1060 6GB (Zotac mini) in an Akitio Thunder 2 case that connects to a mac mini using thunderbolt 2.

The recommended CUDA Toolkit version was the [9.1](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html), but some light algos could be faster with the version 7.5 and 8.0 (like lbry, decred and skein).

# Prerequisites
------------------------------

#Step 1: Install NVidia webdrivers
Your NVidia GPU should work with your Mac as eGPU or with your Hackintosh

Use the following guide for the latest instructions on installing the drivers for your eGPU
https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/paged/1/

I used the follwoing for High Sierra 10.13.3:
- Download this script to easy install NVidia's webdriver: https://github.com/vulgo/webdriver.sh
- Open a terminal and cd into the directory this script is downloaded
- Find the your version of MacOS (about this mac -> system overview -> software: e.g. Systemversion macOS 10.13.2 (17C89))
- Find the link for the nvidia web drivers that correspond to your mac's version and copy the link to the webdrivers on https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/paged/1/
- In the previously opened terminal, run the following command (past the link copied from the previous step over the https://...pkg): sudo ./webdriver.sh -cu https://images.nvidia.com/mac/pkg/387/WebDriver-387.10.10.10.25.156.pkg
- This should install the NVidia's webdrivers and will issue a restart, but the GPU is not detected yet. The following step are necessary to detect the GPU succesfully.
- Reboot the mac while holding down both CMD+R key to start into recovery mode (or use WIN + R when using a windows keyboard)
- When the “OS X Utilities” screen appears, pull down the ‘Utilities’ menu at the top of the screen instead, and choose “Terminal”
- Type the following command to unable installing unsigned drivers into the terminal then hit return:
    csrutil enable --without kext
- Find and download the zip package the correspond to your MacOS's version on https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/paged/1/ (e.g.: [nvidia-egpu-v6.zip](https://cdn.egpu.io/wp-content/uploads/wpforo/attachments/71/4376-NVDAEGPUSupport-v6.zip))
- Install the downloaded package by opening it (you may have to go to system preferences -> security and privacy -> open, to start the installation)
- Shutdown and connect you egpu and turn back on.
- You NVidia GPU should not be detected and listed in: about this mac -> system overview -> hardware -> graphics/displays
- If your eGPU is not detected, log out, reconnect your eGPU and login

#Step 2: Install CUDA
This guide is tested using CUDA 9.1. Other versions should work as well.
Download and install CUDA from: http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html

#Step 3: install Homebrew
Open a terminal and run:
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# Compile on MacOS
----------------
When running into trouble, please check the troubleshooting section below
- Open a terminal
- Run the following:
    brew install pkg-config autoconf automake curl openssl llvm cmake
    brew install cliutils/apple/libomp
- git clone this repository by running:
    git clone https://github.com/rooi/ccminer_macos.git
- cd into the directory using:
    cd ccminer_macos
- build using the following:
    ./build.sh
- install using the following:
    sudo ./install
- copy the plist file for optional auto start by running:
    cp com.ccminer.plist ~/Library/LaunchAgents/

# Running ccminer
----------------
After succesfully building and installing, you can run ccminer using the terminal using the following command: ccminer -n

You can run a benchmark using the following command: ccminer --benchmark

Start mining using the following command (replace text in between the < > signs): ccminer -a <ALGORITHM> -o stratum+tcp://<YOUR_MINING_POOL>:<YOUR_MINING_POOLS_PORT> -u <YOUR_WALLET> -p stats
e.g.: ccminer -a x17 -o stratum+tcp://yiimp.eu:3737 -u DABswnAkgKeXKkmzk8wPdM1kyPgV6xf2MT -p stats

Another option is to start ccminer automatically using a launchdaemon. However, it seems that ccminer does not run full speed when starting up this way, so for now, starting up using a terminal is recommended. If you like to use the daemon you can use the template file: com.ccminer.plist and change it to your configuration (pool, port, wallet, and other options). Afterward, copy the file to the launchagents directory using: cp com.ccminer.plist ~/Library/LaunchAgents/com.ccminer.plist
Then load the daemon to enable starting up at load using: launchctl load ~/Library/LaunchAgents/com.ccminer.plist

Troubleshooting
----------------
Error when running ./configure:
./configure: line 6165: syntax error near unexpected token `,'
./configure: line 6165: `LIBCURL_CHECK_CONFIG(, 7.15.2, ,'

Try to solve using a terminal with the following command:
brew link curl --force


