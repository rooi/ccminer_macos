# ccminer_macos

Forked from Tanguy Pruvot (tpruvot) to create a macos version of ccminer to run easily on mac computers with NVidia eGPU's or Hackintosh.
Originally based on Christian Buchner's &amp; Christian H.'s CUDA project, no more active on github since 2014.

Check the [README.txt](README.txt) for the additions

Most work done on ccminer is done by others such as tpruvot so I'll just leave his BTC donation address here: 1AJdfCpLWPNoAMDfHF1wD5y8VgKSSTHxPo (tpruvot)

In case you appriciate the macos specifics, you can use the following:
ETH donation address: 0x8FE5b2E81C265814e653574a4136480Fa294370d (rooi)

A part of the recent algos were originally written by [djm34](https://github.com/djm34) and [alexis78](https://github.com/alexis78)

This variant was tested and built on MacOS (High Sierra) and runs using a NVidia GTX 1060 6GB (Zotac mini) in an Akitio Thunder 2 case that connects to a mac mini using thunderbolt 2.

The recommended CUDA Toolkit version was the [9.1](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html), but some light algos could be faster with the version 7.5 and 8.0 (like lbry, decred and skein).

Prerequisites
------------------------------

Your NVidia GPU should work with your Mac as eGPU or with your Hackintosh

Use the following guide for the latest instructions on installing the drivers for your eGPU
https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/paged/1/

I used the follwoing for High Sierra 10.13.3:
- Download this script to easy install NVidia's webdriver: https://github.com/vulgo/webdriver.sh
- Open a terminal and cd into the directory the this script is downloaded
- Find the your version of MacOS (about this mac -> system overview -> software: e.g. Systemversion macOS 10.13.2 (17C89))
- Find the link for the nvidia web drivers that correspond to your mac's version and copy the link to the webdrivers on https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/paged/1/
- In the previously opened terminal, run the following command (past the link copied from the previous step over the https://...pkg): sudo ./webdriver.sh https://images.nvidia.com/mac/pkg/387/WebDriver-387.10.10.10.25.156.pkg
- This should install the NVidia's webdrivers and will issue a restart, but the GPU is not detected yet. The following step are neccessary to detect the GPU succesfully.
- Reboot the mac while holding down both CMD+R key to start into recovery mode
- When the “OS X Utilities” screen appears, pull down the ‘Utilities’ menu at the top of the screen instead, and choose “Terminal”
- Type the following command to unable installing unsigned drivers into the terminal then hit return: csrutil enable --without kext
- Find and download the zip package the correspond to your MacOS's version on https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/paged/1/ (e.g.: nvidia-egpu-v6.zip https://cdn.egpu.io/wp-content/uploads/wpforo/attachments/71/4376-NVDAEGPUSupport-v6.zip)
- Shutdown and connect you egpu and turn back on.
- If your eGPU is not detected, log out, reconnect your eGPU and login

Compile on MacOS
----------------
TODO


