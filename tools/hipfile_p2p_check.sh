#!/usr/bin/env bash
# Collect what decides whether hipFile can DMA NVMe -> VRAM directly (PCIe P2P).
# Read-only: it changes no setting. Parts marked [sudo] are skipped without root.
#
#   tools/hipfile_p2p_check.sh            # unprivileged part
#   sudo tools/hipfile_p2p_check.sh       # adds ACS and kernel log
#
# Paste the whole output into the hipFile tracking issue.
set -u

section() { printf '\n=== %s ===\n' "$1"; }
root=0; [ "$(id -u)" -eq 0 ] && root=1

section "kernel"
uname -r
cat /proc/cmdline
config=/boot/config-$(uname -r)
if [ -r "$config" ]; then
    grep -E '^CONFIG_(PCI_P2PDMA|HSA_AMD_P2P|DMABUF_MOVE_NOTIFY|AMD_IOMMU|IOMMU_DEFAULT_[A-Z_]+)=' "$config"
else
    echo "no readable $config"
fi

section "GPUs and NVMe devices"
gpus=$(lspci -D -d 1002: | awk '/VGA|Display|Processing/ {print $1}')
nvmes=$(lspci -D | awk '/Non-Volatile memory/ {print $1}')
for dev in $gpus $nvmes; do
    lspci -D -s "$dev"
    path=$(readlink -f "/sys/bus/pci/devices/$dev")
    echo "  sysfs path: $path"
    group=$(readlink "/sys/bus/pci/devices/$dev/iommu_group" 2>/dev/null)
    echo "  iommu group: ${group##*/}"
    if [ -d "/sys/bus/pci/devices/$dev/p2pmem" ]; then
        echo "  p2pmem: size=$(cat /sys/bus/pci/devices/$dev/p2pmem/size 2>/dev/null)" \
             "available=$(cat /sys/bus/pci/devices/$dev/p2pmem/available 2>/dev/null)" \
             "published=$(cat /sys/bus/pci/devices/$dev/p2pmem/published 2>/dev/null)"
    fi
    lspci -s "$dev" -vv 2>/dev/null | grep -E 'LnkSta:|Region 0|Region 2|Physical Slot' | sed 's/^/  /'
done

section "PCIe tree"
lspci -tv

section "root ports above GPU and NVMe: ACS [sudo]"
if [ "$root" -eq 1 ]; then
    for dev in $gpus $nvmes; do
        path=$(readlink -f "/sys/bus/pci/devices/$dev")
        for hop in $(echo "$path" | tr '/' '\n' | grep -E '^[0-9a-f]{4}:'); do
            [ "$hop" = "$dev" ] && continue
            echo "-- $hop (above $dev)"
            lspci -s "$hop" -vvv | grep -E 'ACSCap|ACSCtl' | sed 's/^/  /'
        done
    done
else
    echo "skipped: run with sudo to read ACS capability and control"
fi

section "kernel log: iommu / p2p / amdgpu BAR [sudo]"
if [ "$root" -eq 1 ]; then
    dmesg | grep -i -E 'p2pdma|p2p|AMD-Vi|iommu: Default|resizable|BAR|kfd' | tail -60
else
    echo "skipped: run with sudo"
fi

section "hipFile library"
for dir in ${ROCM_PATH:-/opt/rocm}/lib $(dirname "$(command -v hipcc 2>/dev/null)")/../lib; do
    ls "$dir"/libhipfile.so* 2>/dev/null
done
echo "Note: hipFile 0.5.0 always reports n_p2p_reads = 0 (field is zero-filled)."
echo "Compare a run with HIPFILE_FORCE_COMPAT_MODE=true against the default to see"
echo "whether the fastpath is faster than the bounce-buffer path."
