# Temporary patch for LibTok encodeTokenIds recursion bug

set(LIBTOK_FILE_TO_PATCH "${LIBTOK_ROOTDIR}/source/token_id_converter.cpp")

if(EXISTS "${LIBTOK_FILE_TO_PATCH}")
    message(STATUS "Patching LibTok bug in ${LIBTOK_FILE_TO_PATCH}")

    file(READ "${LIBTOK_FILE_TO_PATCH}" file_content)

    # Replace the buggy recursive call with the qualified one
    string(REPLACE
        "encodeTokenIds(tokSeq);"
        "this->TokenIdConverterBase::encodeTokenIds(tokSeq); // [TEMP PATCH APPLIED]"
        file_content "${file_content}"
    )

    file(WRITE "${LIBTOK_FILE_TO_PATCH}" "${file_content}")
    message(STATUS "→ Applied temporary fix to TokenIdConverterBase::encodeTokenIds")

else()
    message(WARNING "LibTok patch skipped — file not found: ${LIBTOK_FILE_TO_PATCH}")
endif()