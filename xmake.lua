add_rules("mode.debug", "mode.release")
includes(".my-xmake/toolchains/myClang.lua")
set_toolchains("myClang")
set_languages("cxx20")


-- Tensorflow headers and libraries (C++ and C libs cannot be linked simultaneously) 
local TF_INSTALL_DIR = os.getenv("TensorFlow_ROOT_DIR") or "/opt/tensorflow"
local PROTOBUF_INSTALL_DIR = os.getenv("Protobuf_ROOT_DIR") or "/opt/protobuf"
add_includedirs(path.join(TF_INSTALL_DIR, "include"))
add_includedirs(path.join(PROTOBUF_INSTALL_DIR, "include"))
add_linkdirs(path.join(TF_INSTALL_DIR, "lib"))
add_links("tensorflow_cc", "tensorflow_framework")
add_shflags("-Wl,--no-as-needed")

if is_mode("debug") then 
   add_cxxflags("-ggdb3")  --most detailed gdb information
   add_cxxflags("-Wconversion")  --throws warnings of implicit conversions
   set_warnings("all")
   set_policy("build.sanitizer.address", true) --enabling detection of memory leaks and other memory related problems
else
    add_cxxflags("-flto=thin -march=native")
end

target("txeo_static")
    set_kind("static")
    add_files("src/*.cpp")
    add_includedirs("include/", { public = true })

target("txeo_shared")
    set_kind("shared")
    add_files("src/*.cpp")
    add_includedirs("include/", { public = true })

target("txeo_example")
    set_kind("binary")
    add_files("examples/*.cpp")
    add_deps("txeo_static")

if is_mode("release") then
target("doxygen")
    set_kind("phony")
    on_build(function (target)
        os.exec("doxygen Doxyfile")
    end)
end

